from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from deepfold.common import residue_constants as rc
from deepfold.losses.procrustes import kabsch
from deepfold.losses.utils import softmax_cross_entropy
from deepfold.utils.rigid_utils import Rigid, Rotation
from deepfold.utils.tensor_utils import masked_mean


def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Computes FAPE loss.

    Args:
        pred_frames: Rigid object of predicted frames.      [*, N_frames]
        target_frames: Rigid object of ground truth frames. [*, N_frames]
        frames_mask: Binary mask for the frames.            [*, N_frames]
        pred_positions: Predicted atom positions.           [*, N_pts, 3]
        target_positions: Ground truth positions.           [*, N_pts, 3]
        positions_mask: Positions mask.                     [*, N_pts]
        length_scale: Length scale by which the loss is divided.
        l1_clamp_distance: Cutoff above which distance errors are disregarded.
        eps: Small value used to regularize denominators.

    Returns:
        FAPE loss tensor.

    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert_jit()[..., None].apply(pred_positions[..., None, :, :])
    local_target_pos = target_frames.invert_jit()[..., None].apply(target_positions[..., None, :, :])

    error_dist = torch.sqrt(torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps)

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    # FP16-friendly averaging.
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def backbone_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: torch.Tensor,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    pred_aff = Rigid.from_tensor_7(traj)
    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )

    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of backbone tensor,
    # normalizes it, and then turns it back to a rotation matrix.
    gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    fape_value = compute_fape(
        pred_frames=pred_aff,
        target_frames=gt_aff[:, None],
        frames_mask=backbone_rigid_mask[:, None],
        pred_positions=pred_aff.get_trans(),
        target_positions=gt_aff[:, None].get_trans(),
        positions_mask=backbone_rigid_mask[:, None],
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )

    if use_clamped_fape is not None:
        unclamped_fape_value = compute_fape(
            pred_frames=pred_aff,
            target_frames=gt_aff[:, None],
            frames_mask=backbone_rigid_mask[:, None],
            pred_positions=pred_aff.get_trans(),
            target_positions=gt_aff[:, None].get_trans(),
            positions_mask=backbone_rigid_mask[:, None],
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
        )

        use_clamped_fape = use_clamped_fape.unsqueeze(-1)

        fape_value = fape_value * use_clamped_fape + unclamped_fape_value * (1 - use_clamped_fape)

    fape_value = torch.mean(fape_value, dim=1)

    return fape_value


def sidechain_loss(
    sidechain_frames: torch.Tensor,
    sidechain_atom_pos: torch.Tensor,
    rigidgroups_gt_frames: torch.Tensor,
    rigidgroups_alt_gt_frames: torch.Tensor,
    rigidgroups_gt_exists: torch.Tensor,
    renamed_atom14_gt_positions: torch.Tensor,
    renamed_atom14_gt_exists: torch.Tensor,
    alt_naming_is_better: torch.Tensor,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    renamed_gt_frames = (1.0 - alt_naming_is_better[..., None, None, None]) * rigidgroups_gt_frames
    renamed_gt_frames = renamed_gt_frames + alt_naming_is_better[..., None, None, None] * rigidgroups_alt_gt_frames

    sidechain_frames = sidechain_frames[:, -1]
    batch_dims = sidechain_frames.shape[:-4]
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)
    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)
    sidechain_atom_pos = sidechain_atom_pos[:, -1]
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(*batch_dims, -1, 3)
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)

    fape_value = compute_fape(
        pred_frames=sidechain_frames,
        target_frames=renamed_gt_frames,
        frames_mask=rigidgroups_gt_exists,
        pred_positions=sidechain_atom_pos,
        target_positions=renamed_atom14_gt_positions,
        positions_mask=renamed_atom14_gt_exists,
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        eps=eps,
    )

    return fape_value


def fape_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    backbone_clamp_distance: float,
    backbone_loss_unit_distance: float,
    backbone_weight: float,
    sidechain_clamp_distance: float,
    sidechain_length_scale: float,
    sidechain_weight: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    backbone_loss_value = backbone_loss(
        backbone_rigid_tensor=batch["backbone_rigid_tensor"],
        backbone_rigid_mask=batch["backbone_rigid_mask"],
        traj=outputs["sm_frames"],
        use_clamped_fape=batch.get("use_clamped_fape", None),
        clamp_distance=backbone_clamp_distance,
        loss_unit_distance=backbone_loss_unit_distance,
        eps=eps,
    )

    sidechain_loss_value = sidechain_loss(
        sidechain_frames=outputs["sm_sidechain_frames"],
        sidechain_atom_pos=outputs["sm_positions"],
        rigidgroups_gt_frames=batch["rigidgroups_gt_frames"],
        rigidgroups_alt_gt_frames=batch["rigidgroups_alt_gt_frames"],
        rigidgroups_gt_exists=batch["rigidgroups_gt_exists"],
        renamed_atom14_gt_positions=batch["renamed_atom14_gt_positions"],
        renamed_atom14_gt_exists=batch["renamed_atom14_gt_exists"],
        alt_naming_is_better=batch["alt_naming_is_better"],
        clamp_distance=sidechain_clamp_distance,
        length_scale=sidechain_length_scale,
        eps=eps,
    )

    fape_loss_value = backbone_loss_value * backbone_weight + sidechain_loss_value * sidechain_weight

    return fape_loss_value


def supervised_chi_loss(
    angles_sin_cos: torch.Tensor,
    unnormalized_angles_sin_cos: torch.Tensor,
    aatype: torch.Tensor,
    seq_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    chi_angles_sin_cos: torch.Tensor,
    chi_weight: float,
    angle_norm_weight: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Torsion Angle Loss.

    Supplementary '1.9.1 Side chain and backbone torsion angle loss':
    Algorithm 27 Side chain and backbone torsion angle loss.

    Args:
        angles_sin_cos: Predicted angles.        [*, N, 7, 2]
        unnormalized_angles_sin_cos:             [*, N, 7, 2]
            The same angles, but unnormalized.
        aatype: Residue indices.                 [*, N]
        seq_mask: Sequence mask.                 [*, N]
        chi_mask: Angle mask.                    [*, N, 7]
        chi_angles_sin_cos: Ground truth angles. [*, N, 7, 2]
        chi_weight: Weight for the angle component of the loss.
        angle_norm_weight: Weight for the normalization component of the loss.

    Returns:
        Torsion angle loss tensor.

    """
    pred_angles = angles_sin_cos[..., 3:, :]
    residue_type_one_hot = F.one_hot(aatype, rc.RESTYPE_NUM + 1)
    chi_pi_periodic = torch.einsum(
        "ijk,kl->ijl",
        residue_type_one_hot.type(angles_sin_cos.dtype),
        angles_sin_cos.new_tensor(rc.CHI_PI_PERIODIC),
    )

    true_chi = chi_angles_sin_cos[:, None]

    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1).unsqueeze(-4)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum((true_chi_shifted - pred_angles) ** 2, dim=-1)
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)

    sq_chi_loss = masked_mean(chi_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3))

    loss = chi_weight * sq_chi_loss

    angle_norm = torch.sqrt(torch.sum(unnormalized_angles_sin_cos**2, dim=-1) + eps)
    norm_error = torch.abs(angle_norm - 1.0)
    angle_norm_loss = masked_mean(seq_mask[..., None, :, None], norm_error, dim=(-1, -2, -3))

    loss = loss + angle_norm_weight * angle_norm_loss

    return loss


def compute_distogram(
    positions: torch.Tensor,
    mask: torch.Tensor,
    min_bin: float = 2.3125,
    max_bin: float = 21.6875,
    num_bins: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    boundaries = torch.linspace(min_bin, max_bin, steps=(num_bins - 1), device=positions.device)
    boundaries = boundaries**2
    positions = positions.float()

    dists = torch.sum((positions[..., :, None, :] - positions[..., None, :, :]) ** 2, dim=-1, keepdim=True).detach()

    true_bins = torch.sum(dists > boundaries, dim=-1)

    mask = mask.float()
    pair_mask = mask[..., :, None] * mask[..., None, :]

    return true_bins, pair_mask


def distogram_loss(
    logits: torch.Tensor,
    pseudo_beta: torch.Tensor,
    pseudo_beta_mask: torch.Tensor,
    min_bin: float = 2.3125,
    max_bin: float = 21.6875,
    num_bins: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:

    true_bins, square_mask = compute_distogram(
        positions=pseudo_beta,
        mask=pseudo_beta_mask,
        min_bin=min_bin,
        max_bin=max_bin,
        num_bins=num_bins,
    )

    errors = softmax_cross_entropy(logits=logits, targets=F.one_hot(true_bins, num_bins))

    # FP16-friendly sum.
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    return mean


def compute_renamed_ground_truth(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    atom14_pred_positions = atom14_pred_positions.float()
    pred_dists = torch.sqrt(
        eps + torch.sum((atom14_pred_positions[..., None, :, None, :] - atom14_pred_positions[..., None, :, None, :, :]) ** 2, dim=-1)
    )

    atom14_gt_positions = batch["atom14_gt_positions"].float()
    gt_dists = torch.sqrt(eps + torch.sum((atom14_gt_positions[..., None, :, None, :] - atom14_gt_positions[..., None, :, None, :, :]) ** 2, dim=-1))

    atom14_alt_gt_positions = batch["atom14_alt_gt_positions"].float()
    alt_gt_dists = torch.sqrt(
        eps + torch.sum((atom14_alt_gt_positions[..., :, None, :, None, :] - atom14_alt_gt_positions[..., None, :, None, :, :]) ** 2, dim=-1)
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

    renamed_atom14_gt_positions = (1.0 - alt_naming_is_better[..., None, None]) * atom14_gt_positions
    renamed_atom14_gt_positions = renamed_atom14_gt_positions + alt_naming_is_better[..., None, None] * atom14_alt_gt_positions

    renamed_atom14_gt_mask = (1.0 - alt_naming_is_better[..., None]) * atom14_gt_exists
    renamed_atom14_gt_mask = renamed_atom14_gt_mask + alt_naming_is_better[..., None] * batch["atom14_alt_gt_exists"].float()

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
    """Calculate RMSD.

    This function doesn't superimpose positions.
    """

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
    """Calculate the optimal superimposition."""

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
