import sys
import os
import glob
import pandas as pd
import argparse
import torch
import re
from types import SimpleNamespace

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

module_path = os.path.join(current_dir, "helper_scripts", "eval", "similarity")
if module_path not in sys.path:
    sys.path.append(module_path)

from helper_scripts.eval.similarity.similarity_eval import run_evaluation
from helper_scripts.eval.similarity.mt.embeddings import ImageEmbeddingExtractor
from eval_pipeline.eval_config import FILE_SIM_CSV, CALC_MEMORIZATION, EVAL_REFERENCE_DIR_UNLABLED, TRAIN_DIR_DEFAULT, LAYERS, MODELS, K_NEIGHBORS, CACHE_DIR, FORCE_RECALC_METRICS

def parse_folder_info(folder_name):
    match = re.search(r'^(.*)_(C\d+)_(N\d+)$', folder_name)
    if match:
        run_id = match.group(1) 
        epoch_str = match.group(2) 
        epoch = str(int(epoch_str.replace('C', '')))
        return run_id, epoch
    return folder_name, "unknown"

def evaluate_all(runs_dir, train_dir):
    real_dir = EVAL_REFERENCE_DIR_UNLABLED 
    print(f"Scanning for generated datasets in {runs_dir}...")
    
    pattern_root = os.path.join(runs_dir, "*", "*_C*_N*", "synthetic")
    pattern_direct = os.path.join(runs_dir, "*_C*_N*", "synthetic")
    
    dirs_root = glob.glob(pattern_root)
    dirs_direct = glob.glob(pattern_direct)
    synth_dirs = list(set(dirs_root + dirs_direct))
    synth_dirs = [d for d in synth_dirs if "_C" in os.path.basename(os.path.dirname(d))]

    if not synth_dirs:
        print(f"No 'synthetic' folders found in {runs_dir}.")
        return

    print(f"Found {len(synth_dirs)} folders to evaluate.")
    
    results_dict = {}
    out_csv_path = os.path.join(runs_dir, FILE_SIM_CSV)
    
    if FORCE_RECALC_METRICS and os.path.exists(out_csv_path):
        print(f"FORCE_RECALC_METRICS ist aktiv: Lösche alte CSV unter {out_csv_path}")
        os.remove(out_csv_path)

    #  Load existing cache to prevent redundant recalculations
    if os.path.exists(out_csv_path):
        print(f"Loading existing metrics from {out_csv_path}...")
        df_existing = pd.read_csv(out_csv_path)
        # Convert to dictionary with 'Folder' as the key
        for _, row in df_existing.iterrows():
            results_dict[row['Folder']] = row.to_dict()

    base_embeddings_refreshed = False

    for synth_path in synth_dirs:
        dataset_root = os.path.dirname(synth_path) 
        folder_name = os.path.basename(dataset_root)
        run_id, epoch = parse_folder_info(folder_name)
        
        #  Initialize if not in cache, otherwise update basics
        if folder_name not in results_dict:
            results_dict[folder_name] = {}
        results_dict[folder_name].update({"Run": run_id, "Epoch": epoch, "Folder": folder_name})

    for model_name in MODELS:
        model_short = model_name.split("/")[-1].split("-")[0] 
        
        for layer in LAYERS:
            # Check if this specific model/layer combination is already computed for ALL folders
            target_key = f"{model_short}_L{layer}"
            is_cached = all(
                target_key in results_dict[os.path.basename(os.path.dirname(p))] 
                for p in synth_dirs
            )
            
           
            if is_cached and not FORCE_RECALC_METRICS:
                print(f"--- Skipping {model_name} (Layer {layer}): Already calculated. ---")
                continue 
            elif is_cached and FORCE_RECALC_METRICS:
                print(f"--- FORCE_RECALC_METRICS is True: Forcing recalculation for {model_name} (Layer {layer}). ---")

            print(f"\n{'='*60}")
            
            temp_cfg = SimpleNamespace(
                model_name=model_name, 
                embedding_layer=layer, 
                device="cuda"
            )
            extractor = ImageEmbeddingExtractor(temp_cfg)
            
            for synth_path in synth_dirs:
                dataset_root = os.path.dirname(synth_path)
                folder_name = os.path.basename(dataset_root)
                print(f"   --> Evaluating {folder_name}...")
                
                args = SimpleNamespace(
                    real_dir=real_dir,
                    synth_dir=synth_path,
                    out_dir=dataset_root, 
                    model=model_name,
                    layer=layer,
                    k=K_NEIGHBORS,
                    top_n=10,
                    cache_dir=CACHE_DIR,
                    train_dir=train_dir if CALC_MEMORIZATION else None,
                    skip_vis=True,
                    force_recalc=FORCE_RECALC_METRICS and not base_embeddings_refreshed,
                )
                
                try:
                    metrics = run_evaluation(args, return_metrics=True, extractor=extractor)
                    base_embeddings_refreshed = True
                    if metrics:
                        # 1. capture all metrics
                        for k, v in metrics.items():
                            if isinstance(v, (int, float, str)) and k != "Model":
                                results_dict[folder_name][f"{model_short}_L{layer}_{k}"] = round(v, 4) if isinstance(v, float) else v

                        # 2. mapping for scripts 03 and 04
                        sim_score_mean = metrics.get(f"Cosine_Sim_Mean_k{K_NEIGHBORS}", 0.0)
                        sim_score_std = metrics.get(f"Cosine_Sim_Std_k{K_NEIGHBORS}", 0.0)
                        intra_real = metrics.get("Intra_Diversity_Real", 0.0)
                        intra_synth = metrics.get("Intra_Diversity_Synth", 0.0)
                        vendi_real = metrics.get("Vendi_Score_Real", 0.0)
                        vendi_synth = metrics.get("Vendi_Score_Synth", 0.0)
                        mem_delta = metrics.get("Memorization_Delta", None)                        
                        results_dict[folder_name][f"{model_short}_L{layer}"] = round(sim_score_mean, 4)
                        results_dict[folder_name][f"{model_short}_L{layer}_Std"] = round(sim_score_std, 4)
                        results_dict[folder_name][f"{model_short}_L{layer}_IntraReal"] = round(intra_real, 4)
                        results_dict[folder_name][f"{model_short}_L{layer}_IntraSynth"] = round(intra_synth, 4)
                        results_dict[folder_name][f"{model_short}_L{layer}_VendiReal"] = round(vendi_real, 2)
                        results_dict[folder_name][f"{model_short}_L{layer}_VendiSynth"] = round(vendi_synth, 2)
                        if mem_delta is not None:
                            results_dict[folder_name][f"{model_short}_L{layer}_Memorization_Delta"] = round(mem_delta, 4)
                        if layer == 5:
                            results_dict[folder_name][f"{model_short}_FID"] = round(metrics.get("FID", 0.0), 2)
                            results_dict[folder_name][f"{model_short}_KID"] = round(metrics.get("KID", 0.0), 4)
                except Exception as e:
                    print(f"!!! Error evaluating {folder_name}: {e}")
            
            del extractor
            torch.cuda.empty_cache()

    summary_results = []
    for folder, data in results_dict.items():
        for model_name in MODELS:
            model_short = model_name.split("/")[-1].split("-")[0]
            
            layer_scores = [data.get(f"{model_short}_L{l}") for l in LAYERS if f"{model_short}_L{l}" in data]
            if layer_scores:
                data[f"{model_short}_Avg"] = round(sum(layer_scores) / len(layer_scores), 4)
                
            intra_s_scores = [data.get(f"{model_short}_L{l}_IntraSynth") for l in LAYERS if f"{model_short}_L{l}_IntraSynth" in data]
            if intra_s_scores:
                data[f"{model_short}_IntraSynth_Avg"] = round(sum(intra_s_scores) / len(intra_s_scores), 4)
                
            intra_r_scores = [data.get(f"{model_short}_L{l}_IntraReal") for l in LAYERS if f"{model_short}_L{l}_IntraReal" in data]
            if intra_r_scores:
                data[f"{model_short}_IntraReal_Avg"] = round(sum(intra_r_scores) / len(intra_r_scores), 4)
                
        summary_results.append(data)

    if summary_results:
        df = pd.DataFrame(summary_results)
        try:
            df["Epoch_Num"] = pd.to_numeric(df["Epoch"])
            df = df.sort_values(by=["Run", "Epoch_Num"])
            df = df.drop(columns=["Epoch_Num"])
        except:
            pass

        out_csv_path = os.path.join(runs_dir, FILE_SIM_CSV)
        df.to_csv(out_csv_path, index=False)
        print(f"\n\nEvaluation Complete. Saved to {out_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_dir', type=str, default=".", help="Root dir containing ddim-run folders OR specific run folder")
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR_DEFAULT, help="Path to training images for memorization check")
    args = parser.parse_args()

    evaluate_all(args.runs_dir, args.train_dir)