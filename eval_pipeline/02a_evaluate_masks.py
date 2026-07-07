import os
import glob
import argparse
import subprocess
import re
import sys
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from eval_pipeline.eval_config import FILE_ANAT_CSV, CONDA_ENV_SAM3

SCRIPT_DIR = PROJECT_ROOT / "helper_scripts" / "eval" / "mask"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

try:
    from mask_metrics import calculate_segmentation_metrics, calculate_downstream_metrics
except ImportError as e:
    print(f" import error of mask_metrics in {SCRIPT_DIR}\n{e}")
    sys.exit(1)
    

# --- CONFIGURATION ---
SAM_SCRIPT_PATH = SCRIPT_DIR / "sam3.py"

SYNTH_SUBDIR = "synthetic"
MASKS_SUBDIR = "masks"
REAL_SUBDIR = "real"
SAM_OUT_SUBDIR = "sam3_predictions"
BINARY_MASK_SUBDIR = "binary"
INSTANCE_MASK_SUBDIR = "instances"

def parse_epoch(folder_name):
    match = re.search(r'C(\d+)', folder_name)
    return int(match.group(1)) if match else -1

def run_sam3_inference(input_dir, output_dir, prompt_id=0):
    bin_dir = os.path.join(output_dir, BINARY_MASK_SUBDIR)
    
    if os.path.exists(bin_dir) and len(glob.glob(os.path.join(bin_dir, "*.png"))) > 5:
        print(f"   [SAM3] Masks found in {os.path.basename(output_dir)}. Skipping inference.")
        return

    print(f"   [SAM3] Running inference on {os.path.basename(input_dir)}...")
    
    if not os.path.exists(SAM_SCRIPT_PATH):
        print(f"ERROR: sam3.py not found at {SAM_SCRIPT_PATH}")
        return

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "conda", "run", "-n", CONDA_ENV_SAM3,
        "python", SAM_SCRIPT_PATH,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--prompt_id", str(prompt_id)
    ]
    
    try:
        subprocess.run(cmd, check=True) 
    except subprocess.CalledProcessError as e:
        print(f"   [Error] SAM3 failed: {e}")

def load_masks_as_dict(mask_dir):
    """
    Loads masks into a dictionary mapping filename (without extension) to the mask array.
    This guarantees accurate pairing even if files are missing or extensions differ.
    """
    files = glob.glob(os.path.join(mask_dir, "*.*"))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    masks_dict = {}
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0]
        try:
            img = Image.open(f)
            
            # Check if it's a multi-page TIFF (Instance Stack)
            if f.lower().endswith(('.tif', '.tiff')) and getattr(img, "n_frames", 1) > 0:
                frames = []
                for i in range(img.n_frames):
                    img.seek(i)
                    arr = np.array(img) > 0
                    frames.append(arr)
                
                stack = np.stack(frames, axis=0)
                
                if stack.shape[0] == 1 and not np.any(stack):
                    masks_dict[basename] = np.empty((0, stack.shape[1], stack.shape[2]), dtype=bool)
                else:
                    masks_dict[basename] = stack
            else:
                # Single image (e.g., GT Mask or Binary Pred)
                arr = np.array(img.convert("L"))
                masks_dict[basename] = arr
                
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return masks_dict

def align_masks(gt_dict, pred_bin_dict, pred_inst_dict):
    """
    Aligns GT and Prediction masks perfectly by filename. 
    Injects empty masks for missing predictions to penalize the model correctly.
    """
    cur_gt = []
    cur_pred_bin = []
    cur_pred_inst = []
    missing_count = 0

    for key, gt_mask in gt_dict.items():
        cur_gt.append(gt_mask)
        
        # Align Binary Masks
        if key in pred_bin_dict:
            cur_pred_bin.append(pred_bin_dict[key])
        else:
            cur_pred_bin.append(np.zeros_like(gt_mask))
            missing_count += 1
            
        # Align Instance Masks
        if key in pred_inst_dict:
            cur_pred_inst.append(pred_inst_dict[key])
        else:
            # Empty 3D stack (0, H, W)
            cur_pred_inst.append(np.empty((0, gt_mask.shape[0], gt_mask.shape[1]), dtype=bool))
            
    return cur_gt, cur_pred_bin, cur_pred_inst, missing_count

def calculate_row_metrics(epoch_label, folder_label, gt_dict, pred_bin_dict, pred_inst_dict, pixel_per_um):
    
    # 1. Align Data strictly by filename
    cur_gt, cur_pred_bin, cur_pred_inst, missing_count = align_masks(gt_dict, pred_bin_dict, pred_inst_dict)
    
    if len(cur_gt) == 0:
        return None

    if missing_count > 0:
        print(f"      [Warning] {missing_count} predicted masks missing. Padded with empty masks to penalize score.")

    try:
        # 2. SKIoU and F1 (Skeletonized, for thin lines)
        seg_metrics_skel, df_skel = calculate_segmentation_metrics(
            gt_masks=cur_gt, 
            pred_masks=cur_pred_bin, 
            use_skeletonized_version=True,
            thresholds=[0.5, 0.75]
        )
        
        # 3. Classic IoU (Area overlap, not skeletonized)
        seg_metrics_classic, df_classic = calculate_segmentation_metrics(
            gt_masks=cur_gt, 
            pred_masks=cur_pred_bin, 
            use_skeletonized_version=False,
            thresholds=[0.5, 0.75]
        )
        
        # 4. Biological Metrics (on separated instances)
        down_metrics = calculate_downstream_metrics(
            gt_masks=cur_gt, 
            pred_masks=cur_pred_inst, 
            pixel_per_micrometer=pixel_per_um, 
            spline_s=0
        )
        
        # Build the final dictionary
        row = {
            "Epoch": epoch_label, 
            "Folder": folder_label,
            # Means
            "SKIoU": seg_metrics_skel.get("SKIoU_mean", 0.0),
            "IoU": seg_metrics_classic.get("IoU/T_mean", 0.0),
            "F1@0.50": seg_metrics_skel.get("F1@0.50", 0.0),
            "F1@0.75": seg_metrics_skel.get("F1@0.75", 0.0),
            "Count n/img": down_metrics.get("Avg Count Pred", 0.0),
            # Standard Deviations
            "SKIoU_std": df_skel["SKIoU_mean"].std() if not df_skel.empty and "SKIoU_mean" in df_skel else 0.0,
            "IoU_std": df_classic["IoU/T_mean"].std() if not df_classic.empty and "IoU/T_mean" in df_classic else 0.0,
            "F1@0.50_std": df_skel["F1@0.50"].std() if not df_skel.empty and "F1@0.50" in df_skel else 0.0,
            "F1@0.75_std": df_skel["F1@0.75"].std() if not df_skel.empty and "F1@0.75" in df_skel else 0.0,
            "Count n/img_std": down_metrics.get("Std Count Pred", 0.0),
            # KL Divergences
            "Length_KL": down_metrics.get("Length_KL", 0.0),
            "Curvature_KL": down_metrics.get("Curvature_KL", 0.0)
        }
        return row
        
    except Exception as e:
        print(f"Error calculating metrics for {folder_label}: {e}")
        return None

def evaluate_run(run_dir, pixel_per_um=9.0):
    
    search_path = os.path.join(run_dir, "*_C*_N*")
    generated_folders = sorted(glob.glob(search_path), key=lambda x: parse_epoch(os.path.basename(x)))
    
    if not generated_folders:
        print(f"No generated folders found in {run_dir}")
        return

    first_folder = generated_folders[0]
    gt_masks_path = os.path.join(first_folder, MASKS_SUBDIR)
    baseline_input_dir = os.path.join(first_folder, REAL_SUBDIR)

    print(f"\n--- Setup: Loading Ground Truth ---")
    if not os.path.exists(gt_masks_path):
        print(f"Error: GT Masks folder not found at {gt_masks_path}")
        return

    gt_dict = load_masks_as_dict(gt_masks_path)
    if not gt_dict:
        print("Error: Could not load GT masks.")
        return
    print(f"Loaded {len(gt_dict)} GT masks. Reusing for Baseline and Checkpoints.")
    
    results = []

    # --- BASELINE EVALUATION (Real Images) ---
    print(f"\n{'='*60}")
    print(f"PROCESSING BASELINE: {baseline_input_dir}")
    print(f"{'='*60}")
    
    if os.path.exists(baseline_input_dir):
        baseline_out_dir = os.path.join(run_dir, "baseline_real_sam_predictions")
        run_sam3_inference(baseline_input_dir, baseline_out_dir)
        
        baseline_pred_bin_path = os.path.join(baseline_out_dir, BINARY_MASK_SUBDIR)
        baseline_pred_inst_path = os.path.join(baseline_out_dir, INSTANCE_MASK_SUBDIR)
        
        baseline_preds_bin_dict = load_masks_as_dict(baseline_pred_bin_path)
        baseline_preds_inst_dict = load_masks_as_dict(baseline_pred_inst_path)
        
        print("   Calculating Baseline Metrics (Real vs GT)...")
        baseline_row = calculate_row_metrics(
            epoch_label="BASELINE",
            folder_label="Real_Test_Images",
            gt_dict=gt_dict,
            pred_bin_dict=baseline_preds_bin_dict,
            pred_inst_dict=baseline_preds_inst_dict,
            pixel_per_um=pixel_per_um
        )
        
        if baseline_row:
            results.append(baseline_row)
            print(f"   -> Baseline SKIoU: {baseline_row.get('SKIoU', 0):.4f}")
    else:
        print(f"Warning: BASELINE_DIR {baseline_input_dir} does not exist. Skipping baseline.")

    # --- CHECKPOINT EVALUATION (Synthetic Images) ---
    print(f"\n{'='*60}")
    print(f"PROCESSING CHECKPOINTS")
    print(f"{'='*60}")

    for gen_folder in generated_folders:
        folder_name = os.path.basename(gen_folder)
        epoch = parse_epoch(folder_name)
        
        synth_img_dir = os.path.join(gen_folder, SYNTH_SUBDIR)
        sam_out_dir = os.path.join(gen_folder, SAM_OUT_SUBDIR)
        pred_bin_dir = os.path.join(sam_out_dir, BINARY_MASK_SUBDIR)
        pred_inst_dir = os.path.join(sam_out_dir, INSTANCE_MASK_SUBDIR)
        
        if not os.path.exists(synth_img_dir):
            continue

        print(f"Processing Epoch {epoch}: {folder_name}")

        run_sam3_inference(synth_img_dir, sam_out_dir)
        
        if not os.path.exists(pred_inst_dir) or not os.path.exists(pred_bin_dir):
            print("   Predictions missing.")
            continue
            
        pred_bin_dict = load_masks_as_dict(pred_bin_dir)
        pred_inst_dict = load_masks_as_dict(pred_inst_dir)

        print("   Calculating Metrics...")
        row = calculate_row_metrics(epoch, folder_name, gt_dict, pred_bin_dict, pred_inst_dict, pixel_per_um)
        if row:
            results.append(row)

    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results)
    full_path = os.path.join(run_dir, FILE_ANAT_CSV)
    df.to_csv(full_path, index=False)
    print(f"\nSaved metrics to {full_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help="Path to the ddim-run folder")
    parser.add_argument('--pixel_per_um', type=float, default=9.0)
    args = parser.parse_args()

    evaluate_run(args.run_dir, args.pixel_per_um)