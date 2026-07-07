import pandas as pd
import numpy as np
from eval_config import *

def normalize_col_robust(series, direction='max', lower_percentile=0.05, upper_percentile=0.95):
    """One-Sided Robust Scaling."""
    fill_value = 0 if direction == 'max' else series.max()
    series = pd.to_numeric(series, errors='coerce').fillna(fill_value)
    
    if direction == 'max':
        p_low, p_high = series.quantile(lower_percentile), series.max()
    else:
        p_low, p_high = series.min(), series.quantile(upper_percentile)
        
    clipped_series = series.clip(lower=p_low, upper=p_high)
    if p_high == p_low:
        return np.ones(len(series)) 
    norm = (clipped_series - p_low) / (p_high - p_low)
    return 1.0 - norm if direction == 'min' else norm

def calculate_composite_scores(df, base_scalars=None):
    """
    Standardized scoring logic used by both Script 03 and Script 04.
    Dynamically handles either global baseline scalars (dict) (for script 03) 
    or row-specific baseline columns (for script 04).
    """
    df = df.copy()
    if base_scalars is None:
        base_scalars = {}

    # --- PART A: Similarity Score ---
    dino_fidelity_total = 0.0
    weight_sum = 0.0
    
    # 1. Cosine Similarity Score
    for layer_key, weight in W_LAYERS.items():
        col_name = f"dinov2_{layer_key}"
        if col_name in df.columns:
            layer_score = normalize_col_robust(df[col_name], 'max')
            dino_fidelity_total += weight * layer_score
            weight_sum += weight
        else:
            print(f"Warning: Column '{col_name}' missing in dataset.")
            
    score_cosine = (dino_fidelity_total / weight_sum) if weight_sum > 0 else 0.0
    
    # 2. Intra-Diversity Difference Score (Closer to 0 difference is better)
    if 'dinov2_IntraSynth_Avg' in df.columns and 'dinov2_IntraReal_Avg' in df.columns:
        df['dinov2_Div_Error'] = (df['dinov2_IntraSynth_Avg'] - df['dinov2_IntraReal_Avg']).abs()
        score_intra_diff = normalize_col_robust(df['dinov2_Div_Error'], 'min') 
    else:
        score_intra_diff = 0.0

    # 3. FID and KID Scores
    score_fid = normalize_col_robust(df['dinov2_FID'], 'min') if 'dinov2_FID' in df.columns else 0.0
    score_kid = normalize_col_robust(df['dinov2_KID'], 'min') if 'dinov2_KID' in df.columns else 0.0

    mem_cols = [c for c in df.columns if 'Memorization_Delta' in c]
    if mem_cols:
        df['Mem_Delta_Abs'] = df[mem_cols].mean(axis=1).abs()
        score_mem = normalize_col_robust(df['Mem_Delta_Abs'], 'min')
    else:
        score_mem = 0.0
    
    # 5. Vendi Score (Deviation from Real Baseline)
    vendi_errors = []
    vendi_weights = []

    for layer_key, weight in W_LAYERS.items():
        synth_col = f"dinov2_{layer_key}_VendiSynth"
        real_col = f"dinov2_{layer_key}_VendiReal"

        if synth_col in df.columns and real_col in df.columns:
            # Absolute Abweichung vom echten Vendi-Score berechnen
            error_col_name = f'Vendi_Error_{layer_key}'
            df[error_col_name] = (df[synth_col] - df[real_col]).abs()

            # Normalisieren (Kleinere Abweichung = Besserer Score)
            layer_score = normalize_col_robust(df[error_col_name], direction='min')

            vendi_errors.append(layer_score * weight)
            vendi_weights.append(weight)

    if vendi_weights and sum(vendi_weights) > 0:
        score_vendi = sum(vendi_errors) / sum(vendi_weights)
    else:
        score_vendi = 0.0
    
    df['Score_Sim'] = (
        W_SIM_COSINE * score_cosine + 
        W_SIM_INTRA_DIFF * score_intra_diff +
        W_SIM_FID * score_fid + 
        W_SIM_KID * score_kid + 
        W_SIM_MEMORIZATION * score_mem +
        W_SIM_VENDI * score_vendi
    )

    # --- PART B: Anatomy Score ---
    
    if USE_BASELINE_DEVIATION:
        # Fetch baselines from columns (Script 04) or fallback to provided scalars (Script 03)
        b_skiou = df['Base_SKIoU'] if 'Base_SKIoU' in df.columns else base_scalars.get('skiou', 0.0)
        b_f1 = df['Base_F1'] if 'Base_F1' in df.columns else base_scalars.get('f1', 0.0)
        b_f1_75 = df['Base_F1_75'] if 'Base_F1_75' in df.columns else base_scalars.get('f1_75', 0.0)
        b_count = df['Base_Count'] if 'Base_Count' in df.columns else base_scalars.get('count', 0.0)
        b_len_kl = df['Base_Length_KL'] if 'Base_Length_KL' in df.columns else base_scalars.get('len_kl', 0.0)
        b_curv_kl = df['Base_Curvature_KL'] if 'Base_Curvature_KL' in df.columns else base_scalars.get('curv_kl', 0.0)

        # Deviation calculations (Lower deviation is better -> 'min')
        df['n_len_kl']  = normalize_col_robust((df['Length_KL'] - b_len_kl).abs(), 'min')
        df['n_curv_kl'] = normalize_col_robust((df['Curvature_KL'] - b_curv_kl).abs(), 'min')
        score_anat_bio = (W_BIO_LEN_KL * df['n_len_kl'] + W_BIO_CURV_KL * df['n_curv_kl'])
        
        df['Score_Anat'] = (W_ANAT_FIDELITY * (
            W_FIDELITY_SKIOU * normalize_col_robust((df['SKIoU'] - b_skiou).abs(), 'min') + 
            W_FIDELITY_DICE  * normalize_col_robust((df['F1@0.50'] - b_f1).abs(), 'min') +
            W_FIDELITY_F1_75 * normalize_col_robust((df['F1@0.75'] - b_f1_75).abs(), 'min') +
            W_FIDELITY_COUNT * normalize_col_robust((df['Count n/img'] - b_count).abs(), 'min')
        ) + W_ANAT_BIO * score_anat_bio)
        
    else:
        # Maximize accuracy, Minimize pure KL
        df['n_len_kl']  = normalize_col_robust(df['Length_KL'], 'min')
        df['n_curv_kl'] = normalize_col_robust(df['Curvature_KL'], 'min')
        score_anat_bio = (W_BIO_LEN_KL * df['n_len_kl'] + W_BIO_CURV_KL * df['n_curv_kl'])
        
        df['Score_Anat'] = (W_ANAT_FIDELITY * (
            W_FIDELITY_SKIOU * normalize_col_robust(df['SKIoU'], 'max') + 
            W_FIDELITY_DICE  * normalize_col_robust(df['F1@0.50'], 'max') +
            W_FIDELITY_F1_75 * normalize_col_robust(df['F1@0.75'], 'max') + 
            W_FIDELITY_COUNT * normalize_col_robust(df['Count n/img'], 'max') # Maximizing count is risky if without baseline, but provided for fallback.
        ) + W_ANAT_BIO * score_anat_bio)

   # --- PART C: Color Score ---
    score_col = 0.0
    
    if W_COL_GLOBAL > 0 and 'Global_Dist' in df.columns:
        score_col += W_COL_GLOBAL * normalize_col_robust(df['Global_Dist'], 'min')
        
    if W_COL_FG > 0 and 'FG_Dist' in df.columns:
        score_col += W_COL_FG * normalize_col_robust(df['FG_Dist'], 'min')
        
    if W_COL_BG > 0 and 'BG_Dist' in df.columns:
        score_col += W_COL_BG * normalize_col_robust(df['BG_Dist'], 'min')
        
    df['Score_Col'] = score_col

    # --- PART D: per-axis scaled scores + Global Score (matches paper eq:agg) ---
    # S_global = geometric mean of the THREE selection axes, each already
    # min-max scaled to [0, 1] with "higher = better": Inter-similarity,
    # Intra-diversity and Colour.
    #
    # Reported but EXCLUDED from S_global:
    #   * FID  -- keeps "improving" into the late mode collapse; including it in
    #             the score selects the wrong (later) checkpoint (epoch 400 vs 290).
    #   * Vendi -- redundant with Intra-diversity (r=0.99), so it adds no
    #             independent signal.
    #   * Mask-fidelity (Score_Anat) / filament statistics -- a parametric model
    #             reproduces them by construction.
    # See sections/method.tex (eq:agg), sections/appendix.tex (app:agg-robustness).
    #
    # The geometric mean drops to exactly 0 if any selection axis collapses, so a
    # strong value on one axis cannot mask a failure on another.
    reported_axes = {
        "Inter":  score_cosine,      # normalize_col_robust(dinov2_L5, 'max')
        "Intra":  score_intra_diff,  # gap to real, scaled 'min'
        "Vendi":  score_vendi,       # reported only (redundant with Intra)
        "FID":    score_fid,         # reported only (tracks the late collapse)
        "Colour": df['Score_Col'],   # per-region CIELAB EMD, scaled 'min'
    }
    SELECTION_AXES = ("Inter", "Intra", "Colour")
    present = {}
    for name, val in reported_axes.items():
        if np.isscalar(val):
            # axis column was missing -> score collapsed to a scalar; skip it.
            print(f"Warning: perceptual axis '{name}' unavailable.")
            continue
        arr = np.clip(np.asarray(val, dtype=float), 0.0, None)
        df[f"Axis_{name}"] = arr           # keep scaled per-axis values for plots/tables
        if name in SELECTION_AXES:
            present[name] = arr

    if present:
        stacked = np.vstack(list(present.values()))
        df['Global_Score'] = np.prod(stacked, axis=0) ** (1.0 / len(present))
    else:
        df['Global_Score'] = 0.0

    return df