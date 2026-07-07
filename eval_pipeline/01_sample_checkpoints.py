import os
import json
import subprocess
import glob
import argparse

"""
Batch Sampling Wrapper
"""

# CONSTANTS Must match training settings!
IMG_SIZE = 256
NUM_CHANNELS = 3 
IMG_DIR = "data/eval_dataset_label/imgs/"
SEG_DIR = "data/eval_dataset_label/masks/"
NUM_CLASSES = 2
MODEL_TYPE = "DDIM"
BATCH_SIZE = 8
SEGMENTATION_GUIDED = True

# check if default (linear) or special scheduler was used for training
def get_scheduler_config(checkpoint_path):
    config_path = os.path.join(checkpoint_path, "scheduler", "scheduler_config.json")
    default_schedule = "linear" 
    if not os.path.exists(config_path):
        return default_schedule
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
            return data.get("beta_schedule", default_schedule)
    except Exception:
        return default_schedule


def run_sampling(runs_base_dir, target_runs, sample_size):
    for dataset_name in target_runs:
        # reconstruct dir name as in main.py
        folder_name = f"{MODEL_TYPE.lower()}-{dataset_name}-{IMG_SIZE}"
        if SEGMENTATION_GUIDED:
            folder_name += "-segguided"
        
        run_path = os.path.join(runs_base_dir, folder_name)
        
        if not os.path.isdir(run_path):
            print(f"Skipping {dataset_name}: Directory not found at {run_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing Run: {folder_name}")
        print(f"{'='*60}")

        # Find checkpoints
        checkpoint_dirs = glob.glob(os.path.join(run_path, "checkpoint-*"))
        try:
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
        except ValueError:
            pass
        
        if not checkpoint_dirs:
            print(f"No checkpoints found in {run_path}")
            continue

        for cp_path in checkpoint_dirs:
            epoch_str = os.path.basename(cp_path).split('-')[-1]
            beta_schedule = get_scheduler_config(cp_path)
            
            # Output naming for the generated images
            sample_name_tag = f"{dataset_name}_C{epoch_str}_N{sample_size}"
            
            #  check if dir already exist, if yes dont sample for this checkpoint
            expected_out_dir = os.path.join(run_path, sample_name_tag)
            if os.path.isdir(expected_out_dir):
                print(f"--> Skipping Epoch {epoch_str}: Output folder '{sample_name_tag}' already exists.")
                continue
    

            print(f"\n--> Sampling Epoch {epoch_str} | Schedule: {beta_schedule}")

            cmd = [
                "python", "main.py",
                "--mode", "sample",
                "--model_type", MODEL_TYPE,
                "--img_size", str(IMG_SIZE),
                "--num_img_channels", str(NUM_CHANNELS),
                "--dataset", dataset_name, # Pass the original name, main.py will rebuild the dir string
                "--eval_batch_size", str(BATCH_SIZE),
                "--eval_sample_size", str(sample_size),
                "--seg_dir", SEG_DIR,
                "--img_dir", IMG_DIR,
                "--segmentation_guided",
                "--num_segmentation_classes", str(NUM_CLASSES),
                "--resume_epoch", epoch_str,
                "--beta_schedule", beta_schedule,
                "--custom_run_name", sample_name_tag
            ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error at {dataset_name} epoch {epoch_str}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_dir', type=str, default=".", help="Base directory where output folders are located")
    parser.add_argument('--target_runs', nargs='+', required=True, help="List of dataset names (e.g. run9_B16)")
    parser.add_argument('--sample_size', type=int, default=200)
    args = parser.parse_args()

    run_sampling(args.runs_dir, args.target_runs, args.sample_size)