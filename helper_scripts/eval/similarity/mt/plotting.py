import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
# NeurIPS paper style (central definition: eval_pipeline/plot_style.py).
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "eval_pipeline")))
try:
    from plot_style import apply_neurips_style as _apply_neurips_style
    _apply_neurips_style()
except Exception:
    pass

""" Copied and adapted from /mario-koddenbrock/microtubule_tracking/mt/plotting/plotting.py   """


logger = logging.getLogger(__name__)

def compute_plotting_colour(ref, synth, cfg):

    sims = cosine_similarity(synth, ref)
    return sims.max(axis=1)

def pca_projection(ref, synth, toy=None):
    pca = PCA(n_components=2)
    ref_2d = pca.fit_transform(ref)
    synth_2d = pca.transform(synth)
    toy_2d = pca.transform(toy) if toy is not None else None
    return ref_2d, synth_2d, toy_2d

def tsne_projection(ref, synth, perplexity=30):
    combined = np.vstack([ref, synth])
    n_samples = len(combined)

    perp = min(perplexity, n_samples - 1) if n_samples > 1 else 1
    
    tsne = TSNE(n_components=2, perplexity=perp, init='pca', learning_rate='auto')
    res = tsne.fit_transform(combined)
    return res[:len(ref)], res[len(ref):]

def plot_2d_projection(ref_2d, synth_2d, colour, toy_2d=None, save_to="plot.png", method_name="PCA", **kwargs):
    plt.figure(figsize=(10, 8))

    plt.scatter(ref_2d[:, 0], ref_2d[:, 1], c='lightgray', label='Real (Reference)', alpha=0.5, s=40)

    sc = plt.scatter(synth_2d[:, 0], synth_2d[:, 1], c=colour, cmap='viridis', label='Diffusion (Synth)', alpha=0.9, s=50, edgecolors='k', linewidth=0.3)
    
    plt.colorbar(sc, label="Max Cosine Similarity to Real")
    plt.title(f"{method_name} Projection of Embeddings")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_to, dpi=300)
    plt.close()

def visualize_embeddings(
    cfg, tuning_cfg, # Parameter signature kept for compatibility, strictly not needed if simplified
    ref_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    toy_data=None, # Ignored for simplification
    output_dir: str = "plots/",
    projection_method: str = "PCA",
):
    logger.info(f"Generating {projection_method} plot...")
    os.makedirs(output_dir, exist_ok=True)
    
    colour = compute_plotting_colour(ref_embeddings, synthetic_embeddings, None)
    
    if projection_method.upper() == "TSNE":
        ref_2d, synth_2d = tsne_projection(ref_embeddings, synthetic_embeddings)
    else:
        ref_2d, synth_2d, _ = pca_projection(ref_embeddings, synthetic_embeddings)
        
    save_path = os.path.join(output_dir, f"projection_{projection_method.lower()}.png")
    
    plot_2d_projection(
        ref_2d, synth_2d, colour, 
        save_to=save_path, 
        method_name=projection_method
    )

def plot_similarity_histogram(scores: np.ndarray, output_dir: str, k: int):
    """
    Plots histigramm of cosine similarity values
    """
    plt.figure(figsize=(8, 5))
    
    plt.hist(scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    mean_score = np.mean(scores)
    plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_score:.4f}')
    
    plt.title(f"Distribution of Cosine Similarity (k={k}-NN)")
    plt.xlabel("Cosine Similarity (Higher is better)")
    plt.ylabel("Count of Synthetic Images")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    save_path = os.path.join(output_dir, "score_histogram.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Histogram saved to {save_path}")