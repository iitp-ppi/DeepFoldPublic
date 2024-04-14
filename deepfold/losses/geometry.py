from typing import Dict, Optional, Tuple, Union

import torch

from deepfold.common import residue_constants as rc
from deepfold.losses.procrustes import kabsch
from deepfold.utils.rigid_utils import Rigid


def compute_lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :]) ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :]) ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff) * all_atom_mask * permute_final_dims(all_atom_mask, (1, 0)) * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=-1))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=-1))

    return score


def compute_fape(
    pred_frames: Frame,
    target_frames: Frame,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :].float(),
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :].float(),
    )

    frames_mask = frames_mask.float()
    positions_mask = positions_mask.float()
    error_dist = torch.sqrt(torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps)

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error *= frames_mask[..., None]
    normed_error *= positions_mask[..., None, :]
    if pair_mask is not None:
        normed_error *= pair_mask

    if pair_mask is not None:
        mask = frames_mask.unsqueeze(-1) * positions_mask.unsqueeze(-2)
        mask *= pair_mask
        norm_factor = mask.sum(dim=(-1, -2))
    else:
        norm_factor = torch.sum(frames_mask, dim=-1) * torch.sum(positions_mask, dim=-1)

    normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)

    return normed_error


def compute_distogram(
    positions: torch.Tensor,
    mask: torch.Tensor,
    min_bin: float = 2.3125,
    max_bin: float = 21.6875,
    num_bins: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        steps=(num_bins - 1),
        device=positions.device,
    )
    boundaries = boundaries**2
    positions = positions.float()

    dists = torch.sum(
        (positions[..., :, None, :] - positions[..., None, :, :]) ** 2,
        dim=-1,
        keepdim=True,
    ).detach()

    mask = mask.float()
    pair_mask = mask[..., :None] * mask[..., None, :]

    return torch.sum(dists > boundaries, dim=-1), pair_mask


def compute_aligned_error(
    pred_affine_tensor: torch.Tensor,
    true_affine_tensor: torch.Tensor,
    affine_mask: torch.Tensor,
    max_bin: int = 31,
    num_bins: int = 64,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_affine = Rigid.from_tensor_4x4(pred_affine_tensor.float())
    true_affine = Rigid.from_tensor_4x4(true_affine_tensor.float())

    def _points(affine: Rigid) -> torch.Tensor:
        pts = affine.get_trans()[..., None, :, :]
        return affine.invert()[..., None].apply(pts)

    sq_diff = torch.sum((_points(pred_affine) - _points(true_affine)) ** 2, dim=-1).detach()

    boundaries = torch.linspace(0, max_bin, steps=(num_bins - 1), device=pred_affine_tensor.device)
    boundaries = boundaries**2

    affine_mask = affine_mask.float()
    pair_mask = affine_mask[..., :, None] * affine_mask[..., None, :]

    return (
        torch.sqrt(sq_diff + eps),
        torch.sum(sq_diff[..., None] > boundaries, dim=-1),
        pair_mask,
    )


def compute_renamed_ground_truth(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    atom14_pred_positions = atom14_pred_positions.float()
    pred_dists = torch.sqrt(
        eps
        + torch.sum(
            (atom14_pred_positions[..., None, :, None, :] - atom14_pred_positions[..., None, :, None, :, :]) ** 2,
            dim=-1,
        )
    )

    atom14_gt_positions = batch["atom14_gt_positions"].float()
    gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (atom14_gt_positions[..., None, :, None, :] - atom14_gt_positions[..., None, :, None, :, :]) ** 2,
            dim=-1,
        )
    )

    atom14_alt_gt_positions = batch["atom14_alt_gt_positions"].float()
    alt_gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (atom14_alt_gt_positions[..., None, :, None, :] - atom14_alt_gt_positions[..., None, :, None, :, :]) ** 2,
            dim=-1,
        )
    )

    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)

    atom14_gt_exists = batch["atom14_gt_exists"].float()
    atom14_atom_is_ambiguous = batch["atom14_atom_is_ambiguous"].float()
    mask = (
        atom14_gt_exists[..., None, :, None]
        * atom14_atom_is_ambiguous[..., None, :, None]
        * atom14_gt_exists[..., None, :, None, :]
        * (1.0 - atom14_atom_is_ambiguous[..., None, :, None, :])
    )

    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))

    fp_type = atom14_pred_positions.dtype
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)

    renamed_atom14_gt_positions = (1.0 - alt_naming_is_better[..., None, None]) * atom14_gt_positions + alt_naming_is_better[
        ..., None, None
    ] * atom14_alt_gt_positions

    renamed_atom14_gt_mask = (1.0 - alt_naming_is_better[..., None]) * atom14_gt_exists + alt_naming_is_better[..., None] * batch[
        "atom14_alt_gt_exists"
    ].float()

    return {
        "alt_naming_is_better": alt_naming_is_better,
        "renamed_atom14_gt_positions": renamed_atom14_gt_positions,
        "renamed_atom14_gt_exists": renamed_atom14_gt_mask,
    }


@torch.jit.script
def compute_rmsd(
    true_atom_pos: torch.Tensor,
    pred_atom_pos: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    nan_to: float = 1e8,
) -> torch.Tensor:
    sq_diff = torch.square(true_atom_pos - pred_atom_pos).sum(dim=-1, keepdim=False)

    if atom_mask is not None:
        sq_diff = sq_diff[atom_mask]

    msd = torch.mean(sq_diff)
    msd = torch.nan_to_num(msd, nan=nan_to)

    return torch.sqrt(msd + eps)


def get_optimal_transform(
    src_atoms: torch.Tensor,
    tgt_atoms: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        mask = src_atoms.new_ones(src_atoms.shape[:-1])
    r, t = kabsch(src_atoms, tgt_atoms, weights=mask)
    return r, t


def kabsch_rmsd(
    true_atom_pos: torch.Tensor,
    pred_atom_pos: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r, x = get_optimal_transform(true_atom_pos, pred_atom_pos, atom_mask)
    aligned_true_atom_pos = true_atom_pos @ r + x
    return compute_rmsd(aligned_true_atom_pos, pred_atom_pos, atom_mask=atom_mask)
