# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited

from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

from deepfold.common import residue_constants as rc
from deepfold.model.alphafold import feats
from deepfold.utils.geometry import Rigid, Rotation
from deepfold.utils.tensor_utils import batched_gather, masked_mean, permute_final_dims, tensor_tree_map, tree_map


def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -1 * torch.sum(labels * nn.functional.log_softmax(logits, dim=-1), dim=-1)


def sigmoid_cross_entory(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits_dtype = logits.dtype
    logits = logits.double()
    labels = labels.double()

    log_p = nn.functional.logsigmoid(logits)
    log_np = nn.functional.logsigmoid(-1 * logits)

    loss = (-1 * labels) * log_p - (1 - labels) * log_np
    loss = loss.to(dtype=logits_dtype)

    return loss


def torsion_angle_loss(a: torch.Tensor, a_gt: torch.Tensor, a_alt_gt: torch.Tenosr) -> torch.Tensor:
    """
    Torsion angle loss.

    Args:
        a: torch.Tensor [*, N, 7, 2]
            Cosine and sine value of angles
        a_gt: torch.Tensor
        a_alt_gt: torch.Tensor

    Returns:
        torch.Tensor
    """
    # [*, 7, 2]
    norm = torch.norm(a, dim=-1)

    # [*, N, 7, 2]
    a = a / norm.unsqueeze(-1)

    # [*, N, 7]
    diff_norm_gt = torch.norm(a - a_gt, dim=-1)
    diff_norm_alt_at = torch.norm(a - a_alt_gt, dim=-1)
    min_diff = torch.minimum(diff_norm_gt**2, diff_norm_alt_at**2)

    # [*]
    loss_torsion = torch.mean(min_diff, dim=(-1, -2))
    loss_angle_norm = torch.mean(torch.abs(norm - 1), dim=(-1, -2))

    angle_weight = 0.02

    return loss_torsion * angle_weight * loss_angle_norm


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat([bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0)
    return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: torch.Tensor,  # [num_bins]
    aligned_distance_error_probs: torch.Tensor,  # [*, N', N, num_bins]
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),  # [*, N', N]
        bin_centers[-1],
    )


def compute_predicted_aligned_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    num_bins: int = 64,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [*, N', N, num_bins] the logits output from
        PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      num_bins: Number of bins
    Returns:
      aligned_confidence_probs: [*, N', N, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [*, N', N] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: [*] the maximum predicted error possible.
    """
    boundaries = torch.linspace(0, max_bin, steps=(num_bins - 1), device=logits.device)

    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)  # [*, N', N, num_bins]
    (
        predicted_aligned_error,
        max_predicted_aligned_error,
    ) = _calculate_expected_aligned_error(
        alignment_confidence_breaks=boundaries,
        aligned_distance_error_probs=aligned_confidence_probs,
    )

    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }


def compute_tm(
    logits: torch.Tensor,  # [*, N, N, num_bins]
    residue_weights: Optional[torch.Tensor] = None,
    max_bin: int = 31,
    num_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])  # [N]
    assert residue_weights.ndim == 1

    boundaries = torch.linspace(0, max_bin, steps=(num_bins - 1), device=logits.device)

    bin_centers = _calculate_bin_centers(boundaries)
    clipped_n = max(torch.sum(residue_weights), 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)  #  [*, N, N, num_bins]

    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)  # [*, N, N]

    normed_residue_mask = residue_weights / (eps + residue_weights.sum())  # [N]
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)  # [*, N]

    weighted = per_alignment * residue_weights  # [*, N]

    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]
