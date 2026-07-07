import argparse
import pandas as pd
import os

from eval_config import *
from scoring import calculate_composite_scores

def load_data(path_sim, path_anat, path_col):
    try:
        sim_df = pd.read_csv(path_sim)
        anat_df = pd.read_csv(path_anat)
        col_df = pd.read_csv(path_col)
        return sim_df, anat_df, col_df
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e.filename}")
        return None, None, None

def evaluate_run(run_dir):
    print(f"\n{'='*56}")
    print(f"EVALUATING RUN: {run_dir}")
    print(f"{'='*56}")
    
    sim_df, anat_df, col_df = load_data(
        os.path.join(run_dir, FILE_SIM_CSV), 
        os.path.join(run_dir, FILE_ANAT_CSV), 
        os.path.join(run_dir, FILE_COL_CSV)
    )
    if sim_df is None: return

    # Extract ALL Baseline values
    baseline_row = anat_df[anat_df['Epoch'] == 'BASELINE']
    base_scalars = {
        'skiou': baseline_row['SKIoU'].values[0] if not baseline_row.empty else 0.0,
        'iou': baseline_row['IoU'].values[0] if (not baseline_row.empty and 'IoU' in baseline_row.columns) else 0.0,
        'f1': baseline_row['F1@0.50'].values[0] if not baseline_row.empty else 0.0,
        'f1_75': baseline_row['F1@0.75'].values[0] if (not baseline_row.empty and 'F1@0.75' in baseline_row.columns) else 0.0,
        'count': baseline_row['Count n/img'].values[0] if (not baseline_row.empty and 'Count n/img' in baseline_row.columns) else 0.0,
        'len_kl': baseline_row['Length_KL'].values[0] if not baseline_row.empty else 0.0,
        'curv_kl': baseline_row['Curvature_KL'].values[0] if not baseline_row.empty else 0.0
    }

    anat_df_clean = anat_df[anat_df['Epoch'] != 'BASELINE'].copy()
    sim_df['Epoch'] = pd.to_numeric(sim_df['Epoch'], errors='coerce')
    anat_df_clean['Epoch'] = pd.to_numeric(anat_df_clean['Epoch'], errors='coerce')
    col_df['Epoch'] = pd.to_numeric(col_df['Epoch'], errors='coerce')

    df = sim_df.merge(anat_df_clean, on='Epoch').merge(col_df, on='Epoch')

    # Calculate scores based on config
    df = calculate_composite_scores(df, base_scalars=base_scalars)
    
    mode_str = "Anatomy: Baseline Deviation" if USE_BASELINE_DEVIATION else "Anatomy: Maximization"

    # Rank and save
    ranking = df.sort_values(by='Global_Score', ascending=False)
    ranking.to_csv(os.path.join(run_dir, FILE_OUTPUT_CSV), index=False)
    
    best_epoch = ranking.iloc[0]['Epoch']
    best_score = ranking.iloc[0]['Global_Score']
    
    print(f"Mode: {mode_str}")
    print(f"Best Epoch selected: {int(best_epoch)} (Global Score: {best_score:.4f})")
    print(f"Ranking saved to '{FILE_OUTPUT_CSV}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    evaluate_run(args.run_dir)