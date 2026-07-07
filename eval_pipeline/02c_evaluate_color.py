import sys
import os
import glob
import re
import logging
import pandas as pd
import numpy as np
import cv2
from scipy.stats import wasserstein_distance

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

color_path = os.path.join(os.getcwd(), "helper_scripts", "eval", "color")
if color_path not in sys.path:
    sys.path.append(color_path)
from helper_scripts.eval.color.extract_color_stats import ColorExtractor

from eval_pipeline.eval_config import FILE_COL_CSV, EVAL_REFERENCE_DIR_UNLABLED, EVAL_REFERENCE_DIR_LABLED, EVAL_REFERENCE_DIR_LABLED_MASKS
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def parse_epoch(folder_name):
    """Extracts the epoch number (C_xxxx) from the folder name."""
    match = re.search(r'C(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return -1

def get_folder_distributions(extractor, img_dir, mask_dir=None):
    """
    Runs the ColorExtractor on a folder and returns the SUMMED histograms 
    for all images to represent the dataset-level distribution.
    """
    if not os.path.exists(img_dir):
        logger.warning(f"Directory not found: {img_dir}")
        return None

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    img_files = [f for f in img_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    aggregated_hists = None
    
    for img_path in img_files:
        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        img = extractor._load_image(img_path)
        if img is None: continue

        mask = None
        
        if mask_dir and os.path.exists(mask_dir):
            mask_candidates = glob.glob(os.path.join(mask_dir, f"{name_no_ext}.*"))
            # Filter out the actual image to prevent self-masking
            mask_candidates = [m for m in mask_candidates if os.path.abspath(m) != os.path.abspath(img_path)]
            
            if mask_candidates:
                mask = extractor._load_mask(mask_candidates[0])

        # Get Histograms (256 bins per channel)
        stats = extractor.process_image(img, mask, return_histograms=True)
        
        # Initialize the aggregator dictionary on the first valid image
        if aggregated_hists is None:
            aggregated_hists = {k: np.zeros_like(v, dtype=np.float64) for k, v in stats.items()}
            
        # Accumulate histograms
        for k, v in stats.items():
            aggregated_hists[k] += v
            
    return aggregated_hists

def calculate_real_baselines(extractor):
    """
    Calculates the combined real baseline histograms from the global datasets.
    Global color is taken from the large unlabeled dataset.
    FG/BG color is taken from the smaller labeled dataset.
    """
    logger.info("--- Calculating Global Real Baseline Distributions ---")
    
    # 1. Global Baseline (Unlabeled Data)
    unlabeled_dir = str(EVAL_REFERENCE_DIR_UNLABLED)
    logger.info(f"Extracting Global distributions from: {unlabeled_dir}")
    global_hists = get_folder_distributions(extractor, unlabeled_dir, mask_dir=None)
    
    # 2. FG/BG Baseline (Labeled Data)
    labeled_img_dir = str(EVAL_REFERENCE_DIR_LABLED)
    labeled_mask_dir = str(EVAL_REFERENCE_DIR_LABLED_MASKS)
    
    logger.info(f"Extracting FG/BG distributions from: {labeled_img_dir} & {labeled_mask_dir}")
    labeled_hists = get_folder_distributions(extractor, labeled_img_dir, labeled_mask_dir)
    
    # Combine into a single reference dictionary
    combined_hists = {}
    
    if global_hists is not None:
        combined_hists["Global_L"] = global_hists["Global_L"]
        combined_hists["Global_A"] = global_hists["Global_A"]
        combined_hists["Global_B"] = global_hists["Global_B"]
    elif labeled_hists is not None:
        logger.warning("Unlabeled dataset failed. Falling back to labeled dataset for Global color.")
        combined_hists["Global_L"] = labeled_hists["Global_L"]
        combined_hists["Global_A"] = labeled_hists["Global_A"]
        combined_hists["Global_B"] = labeled_hists["Global_B"]
    else:
        logger.error("Failed to calculate ANY real global baseline.")
        return None

    if labeled_hists is not None:
        combined_hists["FG_L"] = labeled_hists["FG_L"]
        combined_hists["FG_A"] = labeled_hists["FG_A"]
        combined_hists["FG_B"] = labeled_hists["FG_B"]
        combined_hists["BG_L"] = labeled_hists["BG_L"]
        combined_hists["BG_A"] = labeled_hists["BG_A"]
        combined_hists["BG_B"] = labeled_hists["BG_B"]
    else:
        logger.error("Failed to calculate real FG/BG baseline.")
        return None
        
    logger.info("Baseline calculation successful.")
    return combined_hists

def main(run_root_dir, output_csv=FILE_COL_CSV):
    extractor = ColorExtractor()
    
    search_pattern = os.path.join(run_root_dir, "*_C*_N*")
    checkpoint_folders = sorted(glob.glob(search_pattern))
    
    if not checkpoint_folders:
        logger.error(f"No checkpoint folders found in {run_root_dir}")
        return

    logger.info(f"Found {len(checkpoint_folders)} checkpoints.")
    
    # --- A. Calculate Real Distributions (ONCE from Global Datasets) ---
    real_hists = calculate_real_baselines(extractor)
    if real_hists is None:
        logger.error("Could not calculate baseline distributions. Exiting.")
        return
    
    results = []
    # 256 bins representing the values 0-255 in OpenCV LAB
    bins = np.arange(256)
    
    for cp_folder in checkpoint_folders:
        epoch = parse_epoch(os.path.basename(cp_folder))
        synth_dir = os.path.join(cp_folder, "synthetic")
        mask_dir = os.path.join(cp_folder, "masks")
        
        # --- B. Calculate Synthetic Distributions ---
        logger.info(f"Processing Epoch {epoch}...")
        if not os.path.exists(synth_dir):
            logger.warning(f"Synthetic folder missing for {cp_folder}, skipping.")
            continue
            
        synth_hists = get_folder_distributions(extractor, synth_dir, mask_dir)
        if synth_hists is None:
            continue

        # --- C. Compare (Wasserstein Distance) ---
        row = {"Epoch": epoch, "Path": cp_folder}
        
        total_error = 0
        valid_regions = 0
        
        for prefix in ["Global", "FG", "BG"]:
            channel_dists = []
            
            for channel in ["L", "A", "B"]:
                key = f"{prefix}_{channel}"
                u_weights = real_hists.get(key)
                v_weights = synth_hists.get(key)
                
                # Check if the region exists in the dataset (sum > 0) to avoid NaN errors
                if u_weights is None or v_weights is None or np.sum(u_weights) == 0 or np.sum(v_weights) == 0:
                    continue
                    
                # Calculate Earth Mover's Distance
                # 'bins' acts as the values (0-255), 'weights' act as the counts
                wd = wasserstein_distance(bins, bins, u_weights=u_weights, v_weights=v_weights)
                channel_dists.append(wd)
            
            if len(channel_dists) == 3:
                # Average distance across L, A, and B channels for this region
                avg_dist = np.mean(channel_dists)
                row[f"{prefix}_Dist"] = avg_dist
                total_error += avg_dist
                valid_regions += 1
            else:
                row[f"{prefix}_Dist"] = np.nan

        row["Total_Color_Distance"] = total_error / valid_regions if valid_regions > 0 else np.nan
        results.append(row)

    # --- D. Save and Summarize ---
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="Total_Color_Distance", ascending=True)
        
        out_path = os.path.join(run_root_dir, output_csv)
        df_results.to_csv(out_path, index=False)
        
        best_epoch = df_results.iloc[0]
        logger.info("-" * 40)
        logger.info(f"Evaluation finished. Results saved to {out_path}")
        logger.info(f"BEST EPOCH: {int(best_epoch['Epoch'])} (Wasserstein Distance: {best_epoch['Total_Color_Distance']:.4f})")
        logger.info("-" * 40)
        print(df_results[["Epoch", "Total_Color_Distance", "FG_Dist", "BG_Dist"]].to_string(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the DDIM run folder")
    args = parser.parse_args()
    
    main(args.run_dir)