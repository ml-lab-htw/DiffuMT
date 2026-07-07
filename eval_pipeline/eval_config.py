import os
from pathlib import Path
"""
Configuration file for the Microtubule Evaluation Pipeline.
Centralizes all metric weights, evaluation strategies, and filenames 
to ensure scientific reproducibility across all runs.
"""

# ==========================================
# 1. I/O FILENAMES & PATHS & Envs
# ==========================================

PROJECT_ROOT= Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

# Reference datasets (Real images not seen during training)
EVAL_REFERENCE_DIR_LABLED       = DATA_DIR / "data_cp_cluster" / "DATA_FOLDER" / "test"
EVAL_REFERENCE_DIR_LABLED_MASKS = DATA_DIR / "data_cp_cluster" / "MASK_FOLDER" / "all" / "test"
EVAL_REFERENCE_DIR_UNLABLED     = DATA_DIR / "eval_dataset_img" / "imgs"

TRAIN_DIR_DEFAULT = DATA_DIR / "data_cp_cluster"/ "DATA_FOLDER" / "train"

# Expected output filenames for step 2(saved inside the respective run folder)
FILE_SIM_CSV  = 'metrics_similarity.csv'
FILE_ANAT_CSV = 'metrics_anatomy.csv'
FILE_COL_CSV  = 'metrics_color.csv'

#Special env for Mask generation Step 02a
CONDA_ENV_SAM3 = "sam3"

# Output filenames for Step 03
FILE_OUTPUT_CSV  = 'best_epoch_ranking.csv'
FILE_OUTPUT_PLOT = 'best_epoch_trajectory.png'

FORCE_RECALC_METRICS = True
# ==========================================
# 2. GLOBAL WEIGHTS (Must sum to 1.0)
# ==========================================
W_GLOBAL_ANAT = 0.4  # Anatomy is the primary constraint for segmentation tasks.
W_GLOBAL_SIM  = 0.4  # Texture/Realism ensures the domain gap is minimized.
W_GLOBAL_COL  = 0.2 # Color distribution acts as a basic sanity check.

# ==========================================
# 3. METRIC-SPECIFIC WEIGHTS
# ==========================================

# --- A. Similarity Weights & Setiings---

# Configuration
MODELS = [
    "facebook/dinov2-base"
]
LAYERS = [5] 
K_NEIGHBORS = 10
CACHE_DIR = "./embedding_cache"
#Toggle for Calibrated L2 Memorization Check (Computationally heavy)
# Env override lets a light first-pass skip it: CALC_MEMORIZATION=0
CALC_MEMORIZATION = os.environ.get("CALC_MEMORIZATION", "1") not in ("0", "false", "False")

W_SIM_MODEL_DINO = 1.0
W_SIM_MODEL_CLIP = 0.0

# Similarity Internal Explicit Weights (Should sum to 1.0)
W_SIM_COSINE       = 0.5  # Weight for Cosine Similarity (Realism)
W_SIM_INTRA_DIFF   = 0.25   # Weight for Intra-Diversity Error (Mode collapse check)
W_SIM_VENDI        = 0.25
W_SIM_FID          = 0.0 # Weight for Frechet Inception Distance
W_SIM_KID          = 0.0  # Weight for Kernel Inception Distance
W_SIM_MEMORIZATION = 0.0  # Weight for Memorization Delta (Train vs Test similarity)

# Shared Layer Weights (Applied to both DINO and CLIP Cosine Sim)
# Logic: Lower layers (L1) capture edges/noise. Mid layers (L7) capture texture.
W_LAYERS = {
    'L5': 1.0
}

# --- B. Anatomy Weights ---

# For 02a mask accuracy taget 1 (best score wins) = False, taget basline (nearest score to basline wins) = True
USE_BASELINE_DEVIATION = True

W_ANAT_FIDELITY = 0.8  # How well does the structure match the Ground Truth mask?
W_ANAT_BIO      = 0.2  # Biological Plausibility (Statistical distribution correctness)

# Fidelity Internal Weights (Should sum to 1.0)
W_FIDELITY_SKIOU = 0.7  # Skeleton IoU is critical for thin structures (Microtubules).
W_FIDELITY_DICE  = 0.2 # Standard Dice/F1 Score (F1@0.50).
W_FIDELITY_F1_75 = 0.1  # Stricter intersection tracking.
W_FIDELITY_COUNT = 0.0  # Difference in predicted object count vs baseline.

# Bio-Plausibility Internal Weights (KL Divergence -> Lower is Better, sum to 1.0)
W_BIO_LEN_KL  = 0.5  # Distribution of microtubule lengths.
W_BIO_CURV_KL = 0.5  # Distribution of microtubule curvature.

# --- C. Color Weights ---
W_COL_GLOBAL = 0.0  # Overall image color accuracy (Set to 1.0 if no masks are available)
W_COL_FG     = 0.5  # Foreground (Microtubule) color accuracy.
W_COL_BG     = 0.5  # Background (Noise/Artifacts) color accuracy.

# ==========================================
# 4. RUN ALIASES FOR PLOTTING
# ==========================================
# Map the run number to a custom readable label. Applied globally to all plots!
RUN_NAMES = {
    "6":  "Standart",
    "7":  "lin·MSE",
    "8":  "cos·MSE",
    "9":  "cos·L1",
    "10": "cos·Hub·aug3",
    "11": "lin·MSE (Low Data)",
    "12": "lin·MSE·SW (Best)",        
    "13": "cos·Hub·aug2·SW (Aug.)",
    "14": "cos·MSE·o1"
}