from typing import Dict, Optional

import numpy as np
from scipy.special import softmax


def bin_edges_np(
    min_bin: float,
    max_bin: float,
    num_bins: int,
) -> np.ndarray:
    bin_edges = np.linspace(
        start=min_bin,
        stop=max_bin,
        num=num_bins,
    )

    return bin_edges


def calculate_bin_centers_np(boundaries: np.ndarray) -> np.ndarray:
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step * 0.5
    bin_centers = np.concatenate(
        [
            bin_centers,
            (bin_centers[-1] + step)[..., None],
        ],
        axis=0,
    )
    return bin_centers


def undigitize(indices: np.ndarray, bins: np.ndarray, right: bool = False) -> np.ndarray:
    if np.any(indices < 0) or np.any(indices >= len(bins)):
        raise ValueError("Indices must be within the range of bins")

    widths = np.diff(bins)

    if right:
        bins = bins - widths

    return bins[indices]


def compute_predicted_distogram(
    logits: np.ndarray,
    min_bin: float = 2.2325,
    max_bin: float = 21.6875,
    num_bins: int = 64,
) -> np.ndarray:
    boundaries = bin_edges_np(min_bin, max_bin, num_bins - 1)
    bin_centers = calculate_bin_centers_np(boundaries)
    probs = softmax(logits, axis=-1)
    predicted_distoram = np.sum(probs * bin_centers, axis=-1)
    return predicted_distoram


def compute_distogram(
    pts: np.ndarray,
    min_bin: float = 2.2325,
    max_bin: float = 21.6875,
    num_bins: int = 64,
) -> np.ndarray:
    # boundaries = bin_edges_np(min_bin, max_bin, num_bins)
    distances = np.sqrt(np.sum((pts[..., None, :, :] - pts[..., :, None, :]) ** 2, axis=-1))
    return distances
