from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from deepfold.common import residue_constants as rc
from deepfold.losses.utils import calculate_bin_centers, softmax_cross_entropy
from deepfold.utils.rigid_utils import Rigid


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=(0.5 * bin_width),
        end=1.0,
        step=bin_width,
        device=logits.device,
    )
    probs = torch.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * (probs.ndim - 1)), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    """Calculate lDDT score."""

    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(eps + torch.sum((all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :]) ** 2, dim=-1))
    dmat_pred = torch.sqrt(eps + torch.sum((all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :]) ** 2, dim=-1))
    dists_to_score = (dmat_true < cutoff) * all_atom_mask * torch.swapdims(all_atom_mask, -2, -1) * (1.0 - torch.eye(n, device=all_atom_mask.device))
    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


def lddt_ca(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    """Calculate lDDT score with only alhpa-carbon."""

    ca_pos = rc.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    return lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )


def plddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    cutoff: float = 15.0,
    num_bins: int = 50,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Calculate plDDT loss."""

    ca_pos = rc.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    score = lddt(all_atom_pred_pos, all_atom_positions, all_atom_mask, cutoff=cutoff, eps=eps).detach()
    bin_index = torch.floor(score * num_bins).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
    lddt_ca_one_hot = F.one_hot(bin_index, num_classes=num_bins)

    errors = softmax_cross_entropy(logits=logits, labels=lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (eps + torch.sum(all_atom_mask, dim=-1))

    # High resolution only
    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))

    return loss


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: torch.Tensor,
    aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = calculate_bin_centers(alignment_confidence_breaks)
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )


def compute_predicted_aligned_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    num_bins: int = 64,
) -> Dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [*, num_res, num_res, num_bins] the logits output from PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      num_bins: Number of bins

    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: [*] the maximum predicted error possible.

    """
    boundaries = torch.linspace(0, max_bin, steps=(num_bins - 1), device=logits.device)

    aligned_confidence_probs = torch.softmax(logits, dim=-1)

    expected_aligned_error = _calculate_expected_aligned_error(
        alignment_confidence_breaks=boundaries,
        aligned_distance_error_probs=aligned_confidence_probs,
    )

    predicted_aligned_error = expected_aligned_error[0]
    max_predicted_aligned_error = expected_aligned_error[1]

    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }


def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    asym_id: Optional[torch.Tensor] = None,
    interface: bool = False,
    max_bin: int = 31,
    num_bins: int = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute TM score from logis."""

    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(
        start=0,
        end=max_bin,
        steps=(num_bins - 1),
        device=logits.device,
    )

    bin_centers = calculate_bin_centers(boundaries)
    clipped_n = max(torch.sum(residue_weights), 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    n = residue_weights.shape[-1]
    pair_mask = residue_weights.new_ones((n, n), dtype=torch.int32)
    if interface and (asym_id is not None):
        if len(asym_id.shape) > 1:
            assert len(asym_id.shape) <= 2
            batch_size = asym_id.shape[0]
            pair_mask = residue_weights.new_ones((batch_size, n, n), dtype=torch.int32)
        pair_mask *= (asym_id[..., None] != asym_id[..., None, :]).to(dtype=pair_mask.dtype)

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (residue_weights[..., None, :] * residue_weights[..., :, None])
    denom = eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
    normed_residue_mask = pair_residue_weights / denom
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights

    argmax = (weighted == torch.max(weighted)).nonzero()[0]

    return per_alignment[tuple(argmax)]


def tm_loss(
    logits: torch.Tensor,
    final_affine_tensor: torch.Tensor,
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    resolution: torch.Tensor,
    max_bin: int = 31,
    num_bins: int = 64,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_affine = Rigid.from_tensor_7(final_affine_tensor)
    backbone_rigid = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    def _points(affine):
        pts = affine.get_trans()[..., None, :, :]
        return affine.invert()[..., None].apply(pts)

    sq_diff = torch.sum((_points(pred_affine) - _points(backbone_rigid)) ** 2, dim=-1)

    sq_diff = sq_diff.detach()

    boundaries = torch.linspace(start=0, end=max_bin, steps=(num_bins - 1), device=logits.device)
    boundaries = boundaries**2
    true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits=logits,
        labels=F.one_hot(true_bins, num_bins),
    )

    square_mask = backbone_rigid_mask[..., None] * backbone_rigid_mask[..., None, :]

    loss = torch.sum(errors * square_mask, dim=-1)
    scale = 0.5  # hack to help FP16 training along
    denom = eps + torch.sum(scale * square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))

    return loss
