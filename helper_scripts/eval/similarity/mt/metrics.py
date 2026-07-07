import logging
import numpy as np
from scipy.linalg import sqrtm
""" Copied and adapted from /mario-koddenbrock/microtubule_tracking/mt/data_generation/optimization/metrics.py"""


logger = logging.getLogger(__name__)

def compute_frechet_distance(synthetic_embeddings: np.ndarray, ref_embeddings: np.ndarray) -> float:
    mu1 = np.mean(ref_embeddings, axis=0)
    sigma1 = np.cov(ref_embeddings, rowvar=False)
    mu2 = np.mean(synthetic_embeddings, axis=0)
    sigma2 = np.cov(synthetic_embeddings, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)

def _poly_kernel(X, Y):
    gamma = 1.0 / X.shape[1] if X.shape[1] > 0 else 1.0
    return (gamma * (X @ Y.T) + 1.0) ** 3

def compute_kid(synthetic_embeddings: np.ndarray, ref_embeddings: np.ndarray) -> float:
    m, n = ref_embeddings.shape[0], synthetic_embeddings.shape[0]
    if m < 2 or n < 2: return 0.0

    k_xx = _poly_kernel(ref_embeddings, ref_embeddings)
    k_yy = _poly_kernel(synthetic_embeddings, synthetic_embeddings)
    k_xy = _poly_kernel(ref_embeddings, synthetic_embeddings)

    term1 = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    term2 = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
    term3 = 2 * k_xy.sum() / (m * n)
    return float(term1 + term2 - term3)