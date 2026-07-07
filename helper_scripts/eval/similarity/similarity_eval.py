"""
Similarity_eval.py calculates Cosine Similarity, FID and KID Scores
The values for these metrics are the embeddings of original and synthetic images
from DINOv2 or CLIP from a specific layer and K-Neigbors    
"""

import argparse
import logging
import os
import json
import torch
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import torch.nn.functional as F

# Import utis (adapted from /mario-koddenbrock/microtubule_tracking/)
from mt.config import EvalConfig
from mt.embeddings import ImageEmbeddingExtractor
from mt.metrics import compute_frechet_distance, compute_kid
from mt.plotting import plot_similarity_histogram 
from vendi_score import vendi

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def save_knn_visuals(synth_files, real_files, scores, nearest_indices, output_dir, n=10, mode="Best"):
    """
    Saves a comparison of best or worst similarity scores to validate the scores with images
    """
    save_dir = os.path.join(output_dir, "visuals", mode)
    os.makedirs(save_dir, exist_ok=True)
    
    # sort the scores array dsc or asc, get n indices of the best/worst
    if mode == "Best":
        sorted_idx = np.argsort(scores)[::-1][:n]
    else:
        sorted_idx = np.argsort(scores)[:n]

    # get score and filepaths with indices    
    for idx in sorted_idx:
        # synth 
        score = scores[idx]
        s_path = synth_files[idx]
        
        # nn in real files
        nn_idx = nearest_indices[idx]
        r_path = real_files[nn_idx]
        
        # load images and show them sinde by side
        try:
            img_s = Image.open(s_path).resize((256, 256)).convert("RGB")
            img_r = Image.open(r_path).resize((256, 256)).convert("RGB")
            
            dst = Image.new('RGB', (512, 256))
            dst.paste(img_s, (0, 0))
            dst.paste(img_r, (256, 0))
            
            fname = f"{score:.4f}_{os.path.basename(s_path)}"
            dst.save(os.path.join(save_dir, fname))
        except Exception as e:
            logger.warning(f"Failed to save visual for {s_path}: {e}")

def get_cached_embeddings(extractor, img_dir, cache_dir, identifier, force_recalc=False):
    """
    Tries to load embeddings from cache. If not found or force_recalc=True, 
    calculates and saves them.
    """
    if not cache_dir:
        return extractor.load_images_and_extract_embeddings(img_dir)

    os.makedirs(cache_dir, exist_ok=True)
    emb_path = os.path.join(cache_dir, f"{identifier}_embeddings.pt")
    files_path = os.path.join(cache_dir, f"{identifier}_files.pkl")

    if not force_recalc and os.path.exists(emb_path) and os.path.exists(files_path):
        logger.info(f"Loading cached embeddings from {emb_path}...")
        embeddings = torch.load(emb_path, weights_only=False) 
        with open(files_path, 'rb') as f:
            files = pickle.load(f)
        return embeddings, files
    else:
        logger.info(f"Computing and caching embeddings for {identifier} (Force Recalc: {force_recalc})...")
        embeddings, files = extractor.load_images_and_extract_embeddings(img_dir)
        
        torch.save(embeddings, emb_path)
        with open(files_path, 'wb') as f:
            pickle.dump(files, f)
        return embeddings, files

def calculate_intra_diversity(embeddings_norm: torch.Tensor):
    """
    Calculates the per-image pairwise cosine similarity and the global mean.
    Returns: (global_mean_float, per_image_array)
    """
    n = embeddings_norm.size(0)
    if n < 2:
        return 0.0, np.array([0.0])
        
    # Calculate full N x N similarity matrix
    sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    
    # Subtract the diagonal (self-similarity) so it doesn't skew the mean
    sim_matrix.fill_diagonal_(0)
    
    # Sum across columns and divide by (n-1) to get the mean similarity per image
    per_image_sim = sim_matrix.sum(dim=1) / (n - 1)
    
    scalar_mean = float(per_image_sim.mean().item())
    
    return scalar_mean, per_image_sim.cpu().numpy()

def calculate_vendi_score(embeddings_norm: torch.Tensor):
    """
    Computes the Vendi Score using the official vendi_score package.
    Vendi Score measures the effective number of unique modes in the dataset.
    Higher is better (more diverse / less mode collapse).
    """

    K = torch.mm(embeddings_norm, embeddings_norm.t())
    

    K_np = K.cpu().numpy()
    

    score = vendi.score_K(K_np)
    
    return float(score)

def run_evaluation(args, return_metrics=False,extractor=None):
    # Get dir name with model, layer and k values
    model_short = args.model.split("/")[-1]
    folder_name = f"eval_{model_short}_L{args.layer}_K{args.k}"

    target_path = os.path.join(args.out_dir, folder_name)

    # 1. Create Config
    cfg = EvalConfig(
        reference_images_dir=args.real_dir,
        synthetic_images_dir=args.synth_dir,
        output_dir=target_path, 
        embedding_layer=args.layer,
        id=folder_name,
        model_name=args.model
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    logger.info(f"Results will be saved to: {cfg.output_dir}")
    
    
    # 2. Calculate embeddings
    if extractor is None:
        logger.info("Instantiating NEW extractor (Fallback)")
        extractor = ImageEmbeddingExtractor(cfg)
    else:
        extractor.config = cfg 
        
    force = getattr(args, 'force_recalc', False)

    logger.info("Extracting REAL embeddings...")
    real_id = f"real_{model_short}_L{args.layer}"
    real_emb, real_files = get_cached_embeddings(extractor, cfg.reference_images_dir, args.cache_dir, real_id, force_recalc=force)    
    
    logger.info("Extracting SYNTH embeddings...")
    synth_emb, synth_files = extractor.load_images_and_extract_embeddings(cfg.synthetic_images_dir)    
    
    model_identifier = f"{args.model.split('/')[-1]}_L{args.layer}_K{args.k}"
    
    metrics = {
        "Model": model_identifier,
        "FID": compute_frechet_distance(synth_emb, real_emb),
        "KID": compute_kid(synth_emb, real_emb)
    }
    logger.info(f"Global Metrics: {metrics}")
    
    # 4. k-NN
    logger.info(f"Running k-NN analysis (k={args.k})...")
    
    # Ensure tensors are on the correct device and L2-normalized for Cosine Similarity
    real_t = torch.tensor(real_emb).to(extractor.device)
    real_t = F.normalize(real_t, p=2, dim=1)
    
    synth_t = torch.tensor(synth_emb).to(extractor.device)
    synth_t = F.normalize(synth_t, p=2, dim=1)

    # --- 1. Global Metrics (FID/KID) ---
    model_identifier = f"{args.model.split('/')[-1]}_L{args.layer}_K{args.k}"
    metrics = {
        "Model": model_identifier,
        "FID": compute_frechet_distance(synth_emb, real_emb),
        "KID": compute_kid(synth_emb, real_emb)
    }
    
    # --- 2. Cross-Similarity: Synth vs. Real Test (Quality/Fidelity) ---
    # How closely does the generated distribution match the real distribution?
    sim_matrix_cross = torch.mm(synth_t, real_t.t()) 
    topk_vals, topk_inds = torch.topk(sim_matrix_cross, k=args.k, dim=1)
    
    mean_scores_test = topk_vals.mean(dim=1).cpu().numpy()
    nearest_indices = topk_inds[:, 0].cpu().numpy() 

    metrics[f"Cosine_Sim_Mean_k{args.k}"] = float(np.mean(mean_scores_test))
    metrics[f"Cosine_Sim_Std_k{args.k}"] = float(np.std(mean_scores_test))
    
    # 1. Calc Mean and arr
    real_intra_mean, real_intra_array = calculate_intra_diversity(real_t)
    synth_intra_mean, synth_intra_array = calculate_intra_diversity(synth_t)
    
    metrics["Vendi_Score_Real"] = calculate_vendi_score(real_t)
    metrics["Vendi_Score_Synth"] = calculate_vendi_score(synth_t)

    metrics["Intra_Diversity_Real"] = real_intra_mean
    metrics["Intra_Diversity_Synth"] = synth_intra_mean
    
    # 3. save for plotting 
    pd.DataFrame({"score": real_intra_array}).to_csv(os.path.join(cfg.output_dir, "intra_scores_real.csv"), index=False)
    
    logger.info(f"Baseline Diversity (Real): {metrics['Intra_Diversity_Real']:.4f}")
    logger.info(f"Model Diversity (Synth):   {metrics['Intra_Diversity_Synth']:.4f}")

    if args.train_dir:
        logger.info("Extracting TRAINING embeddings for memorization check...")
        try:
            train_id = f"train_{args.model.split('/')[-1]}_L{args.layer}"
            
            train_emb, _ = get_cached_embeddings(extractor, args.train_dir, args.cache_dir, train_id, force_recalc=force)
            train_t = torch.tensor(train_emb).to(extractor.device)
            train_t = F.normalize(train_t, p=2, dim=1)
            
            sim_matrix_train = torch.mm(synth_t, train_t.t())
            topk_vals_train, _ = torch.topk(sim_matrix_train, k=args.k, dim=1)
            
            # 1. Get the single nearest neighbor (k=1) cosine similarity for Train and Test
            max_cos_train = topk_vals_train[:, 0]
            max_cos_test = topk_vals[:, 0] # topk_vals is from the test set evaluation earlier

            # 2. Calibrate: How much more similar is the image to the Train set than the Test set?
            # Higher cosine sim means closer. If max_cos_train > max_cos_test, the model is overfitting.
            calibrated_memorization = max_cos_train - max_cos_test
            
            metrics["Memorization_Delta"] = float(calibrated_memorization.mean().item())
            
        except Exception as e:
            logger.warning(f"Could not perform training comparison: {e}")
    
    mean_scores = mean_scores_test

    with open(os.path.join(cfg.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    
    # csv export with fn, cosine similarity and nn
    df = pd.DataFrame({
        "filename": [os.path.basename(f) for f in synth_files],
        "score": mean_scores,
        "intra_similarity": synth_intra_array,
        "nearest_real_image": [os.path.basename(real_files[i]) for i in nearest_indices]
    })
    df.to_csv(os.path.join(cfg.output_dir, "scores.csv"), index=False)
    
    #  Save worst best n images and histogramm
    if not args.skip_vis:
        logger.info("Saving Best/Worst visuals...")
        save_knn_visuals(synth_files, real_files, mean_scores, nearest_indices, cfg.output_dir, n=args.top_n, mode="Best")
        save_knn_visuals(synth_files, real_files, mean_scores, nearest_indices, cfg.output_dir, n=args.top_n, mode="Worst")

        plot_similarity_histogram(mean_scores, cfg.output_dir, k=args.k)

    logger.info("Evaluation complete.")

    if return_metrics:
        return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True, help="Path to ground truth images")
    parser.add_argument("--synth_dir", required=True, help="Path to synthetic images")
    parser.add_argument("--out_dir", default=f"")
    
    parser.add_argument("--model", default="facebook/dinov2-base", help="e.g. facebook/dinov2-base or openai/clip-vit-base-patch32")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index (default: last)")
    parser.add_argument("--k", type=int, default=1, help="k-Neighbors for analysis")
    parser.add_argument("--top_n", type=int, default=10, help="Number of images to save")
    parser.add_argument("--train_dir", default=None, help="Optional: Path to training images to check for memorization/overfitting")
    parser.add_argument("--cache_dir", default="./embedding_cache", help="Directory to store real/train embeddings")
    parser.add_argument("--skip_vis", action="store_true", help="Skip PCA/TSNE generation for speed")   

    args = parser.parse_args()
    run_evaluation(args)