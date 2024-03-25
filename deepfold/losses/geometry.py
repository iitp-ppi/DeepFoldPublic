from typing import Optional, Tuple, Union

import torch

from deepfold.common import residue_constants as rc
from deepfold.losses.procrustes import kabsch
from deepfold.utils.rigid_utils import Rigid

# compute_lddt

# compute_fape


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


# compute_renamed_ground_truth


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
