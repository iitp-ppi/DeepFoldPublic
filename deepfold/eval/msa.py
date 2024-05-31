import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_neff_v1(
    msa: np.ndarray,
    cutoff: float = 0.62,
    eps: float = 1e-6,
) -> float:
    # assert cutoff > 0.0
    y = pdist(msa, metric="hamming")
    d = squareform(y)
    w = d > cutoff
    neff = np.sum(1.0 / (w.sum(axis=0) + eps))
    return neff


def compute_neff_v2(
    msa: np.ndarray,
    cutoff: float = 0.62,
    eps: float = 1e-10,
) -> float:
    theta = 1.0 - cutoff
    assert theta > 0.0
    y = pdist(msa, metric="hamming")
    d = squareform(y)
    w = 1.0 / (1.0 + np.sum(d < theta, 0))
    neff = np.log2(eps + np.sum(w))
    return neff
