# Copyright 2024 DeepFold Team


"""Embed input features."""


from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from deepfold.common import residue_constants as rc
from deepfold.utils.geometry import Rigid
from deepfold.utils.tensor_utils import batched_gather, one_hot

TensorDict = Dict[str, torch.Tensor]


def pseudo_beta_fn(
    aatype: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_masks: Optional[torch.Tensor],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""
    Extracts pseudo-beta atom positions from amino acid types and atom positions.

    This function determines the position of pseudo-beta atoms for a given amino acid sequence.
    For glycine, which lacks a beta carbon (CB), the alpha carbon (CA) position is used instead.
    If atom masks are provided, the function also returns a mask indicating the presence of pseudo-beta atoms.

    Args:
        aatype: [..., N]
            A tensor containing amino acid types encoded as integers.
        all_atom_positions: [..., N, 37, 3]
            A tensor containing the positions of all atoms.
        all_atom_masks: [..., N, 37]
            An optional tensor indicating the presence (1) or absence (0) of each atom type. Default is None.

    Returns:
        A single tensor containing the positions of pseudo-beta atoms if `all_atom_masks` is None.
        Otherwise, returns a tuple of two tensors: the first for pseudo-beta positions and the second for their masks.
    """
    # Determine whether each amino acid is glycine, which lacks a CB atom
    is_gly = aatype == rc.restype_order["G"]

    # Indexes for alpha carbon (CA) and beta carbon (CB) atoms
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]

    # Use CA position for glycine, CB position otherwise.
    pseudo_beta = torch.where(
        is_gly[..., None].expand(*((-1,) * len(is_gly.shape)), 3),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    # If atom masks are provided, calculate the mask for pseudo-beta atoms
    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_masks[..., ca_idx],
            all_atom_masks[..., cb_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta  # [..., N, 3]


def atom14_to_atom37(atom14: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
    r"""
    Converts atom representations from a 14-atom format to a 37-atom format.

    Args:
        atom14: [..., 14, *]
            A tensor containing atom data in the 14-atom format.
            The tensor's shape should accommodate batch and spatial dimensions as needed.
        batch: A dictionary containing the following key-value pairs:
            - "residx_atom37_to_atom14": The indices mapping 14-atom representations to 37-atom representations.
            - "atom37_atom_exists": A mask indicating whether each atom in the 37-atom format exists.

    Returns:
        A tensor containing the converted atom data in the 37-atom format, with non-existing atoms masked out.
    """
    # Gather the 37-atom data using the provided mapping from the batch dictionary.
    atom37_data = batched_gather(
        atom14,
        batch["residx_atom37_to_atom14"],
        dim=-2,
        num_batch_dims=len(atom14.shape[:-2]),
    )

    # Apply the mask for existing atoms in the 37-atom format.
    atom37_data = atom37_data * batch["atom37_atom_exists"][..., None]

    return atom37_data


def build_template_angle_feat(template_feats: TensorDict) -> Union[torch.Tensor, torch.Tensor]:
    r"""
    Build template angle features for AlphaFold2.

    Requires:
        - template_aatype [N_templ, N_res, 22]
        - template_torsion_angles_sin_cos [N_templ, N_res, 7, 2]
        - template_alt_torsion_angles_sin_cos [N_templ, N_res, 7, 2]
        - template_torsion_angles_mask [N_templ, N_res, 7]

    Returns:
        - template_angle_feat [N_templ, N_res, 57]
        - template_angle_mask [N_templ, N_res, 7]
    """

    template_aatype = template_feats["template_aatype"]
    torsion_angles_sin_cos = template_feats["template_torsion_angles_sin_cos"]
    torsion_angles_mask = template_feats["template_torsion_angles_mask"]

    alt_torsion_angles_sin_cos = template_feats["template_alt_torsion_angles_sin_cos"]
    template_angle_feat = torch.cat(
        [
            one_hot(template_aatype, 22),
            torsion_angles_sin_cos.reshape(*torsion_angles_sin_cos.shape[:-2], 14),
            alt_torsion_angles_sin_cos.reshape(*alt_torsion_angles_sin_cos.shape[:-2], 14),
            torsion_angles_mask,
        ],
        dim=-1,
    )
    template_angle_mask = torsion_angles_mask[..., 2]

    return template_angle_feat, template_angle_mask


def build_template_angle_feat_v2(template_feats: TensorDict) -> Union[torch.Tensor, torch.Tensor]:
    r"""
    Build template angle features for AlphaFold-Multimer.

    Requires
        - template_aatype [N_templ, N_res, 22]
        - template_torsion_angles_sin_cos [N_templ, N_res, 7, 2]
        - template_torsion_angles_mask [N_templ, N_res, 7]

    Returns:
        - template_angle_feat [N_templ, N_res, 34]
        - template_angle_mask [N_templ, N_res, 7]
    """

    template_aatype = template_feats["template_aatype"]
    torsion_angles_sin_cos = template_feats["template_torsion_angles_sin_cos"]
    torsion_angles_mask = template_feats["template_torsion_angles_mask"]

    chi_mask = torsion_angles_mask[..., 3:]
    chi_angles_sin = torsion_angles_sin_cos[..., 3:, 0] * chi_mask
    chi_angles_cos = torsion_angles_sin_cos[..., 3:, 1] * chi_mask
    template_angle_feat = torch.cat(
        [
            one_hot(template_aatype, 22),
            chi_angles_sin,
            chi_angles_cos,
            chi_mask,
        ],
        dim=-1,
    )
    template_angle_mask = chi_mask[..., 0]

    return template_angle_feat, template_angle_mask


def build_template_pair_feat(
    batch: TensorDict,
    min_bin: float,
    max_bin: float,
    num_bins: int,
    inf: float = 1e8,
) -> torch.Tensor:
    r"""
    Build template pair features for AlphaFold2.

    Requires:
        - template_aatype
        - template_pseudo_beta
        - template_pseudo_beta_mask

    Returns:
        template_pair_feat [N_templ, N_res, N_res, 88]
        - distogram [N_templ, N_res, N_res, 39]
        - template_pseudo_beta_mask [N_templ, N_res, N_res, 1]
        - aatype_one_hot_row [N_templ, N_res, N_res, 22]
        - aatype_one_hot_col [N_templ, N_res, N_res, 22]
        - unit_vector [N_templ, N_res, N_res, 3]
        - backbone_frame_mask [N_templ, N_res, N_res, 1]

    Notes:
        - unit_vector is zero tensor.
        - backbone_frame_mask equals to template_pseudo_beta_mask
    """

    template_mask = batch["template_pseudo_beta_mask"]
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

    tpb = batch["template_pseudo_beta"]
    dgram = torch.sum((tpb[..., None, :] - tpb[..., None, :, :]) ** 2, dim=-1, keepdim=True)
    lower = torch.linspace(min_bin, max_bin, num_bins, device=tpb.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)

    to_concat = [dgram, template_mask_2d[..., None]]

    aatype_one_hot = nn.functional.one_hot(
        batch["template_aatype"],
        rc.restype_num + 2,
    )

    n_res = batch["template_aatype"].shape[-1]
    to_concat.append(aatype_one_hot[..., None, :, :].expand(*aatype_one_hot.shape[:-2], n_res, -1, -1))
    to_concat.append(aatype_one_hot[..., None, :].expand(*aatype_one_hot.shape[:-2], -1, n_res, -1))

    to_concat.append(template_mask_2d.new_zeros(*template_mask_2d.shape, 3))
    to_concat.append(template_mask_2d[..., None])

    act = torch.cat(to_concat, dim=-1)
    act = act * template_mask_2d[..., None]

    return act


def build_template_pair_feat_v2(
    batch: TensorDict,
    min_bin: float,
    max_bin: float,
    num_bins: int,
    multichain_mask_2d: Optional[torch.Tensor] = None,
    eps: float = 1e-20,
    inf: float = 1e8,
) -> List[torch.Tensor]:
    r"""
    Build template pair features for AlphaFold-Multimer.

    Requires:
        - template_aatype
        - template_pseudo_beta
        - template_pseudo_beta_mask
        - template_all_atom_positions
        - template_all_atom_mask

    Returns:
        template_pair_feat: List[torch.Tensor]
        - distogram [N_templ, N_res, N_res, 39]
        - template_pseudo_beta_mask [N_templ, N_res, N_res, 1]
        - aatype_one_hot_row [N_templ, N_res, N_res, 22]
        - aatype_one_hot_col [N_templ, N_res, N_res, 22]
        - unit_vector [N_templ, N_res, N_res, 3]
        - backbone_frame_mask [N_templ, N_res, N_res, 1]
    """

    template_mask = batch["template_pseudo_beta_mask"]
    template_mask_2d = template_mask[..., :, None] * template_mask[..., None, :]
    if multichain_mask_2d is not None:
        template_mask_2d *= multichain_mask_2d

    tpb = batch["template_pseudo_beta"]
    dgram = torch.sum((tpb[..., :, None, :] - tpb[..., None, :, :]) ** 2, dim=-1, keepdim=True)
    lower = torch.linspace(min_bin, max_bin, num_bins, device=tpb.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)
    dgram *= template_mask_2d[..., None]
    to_concat = [dgram, template_mask_2d[..., None]]

    aatype_one_hot = one_hot(
        batch["template_aatype"],
        rc.restype_num + 2,
    )

    n_res = batch["template_aatype"].shape[-1]
    to_concat.append(aatype_one_hot[..., None, :, :].expand(*aatype_one_hot.shape[:-2], n_res, -1, -1))
    to_concat.append(aatype_one_hot[..., :, None, :].expand(*aatype_one_hot.shape[:-2], -1, n_res, -1))

    n, ca, c = [rc.atom_order[a] for a in ["N", "CA", "C"]]
    rigids = Rigid.make_transform_from_reference(
        n_xyz=batch["template_all_atom_positions"][..., n, :],
        ca_xyz=batch["template_all_atom_positions"][..., ca, :],
        c_xyz=batch["template_all_atom_positions"][..., c, :],
        eps=eps,
    )
    points = rigids.get_trans()[..., None, :, :]
    rigid_vec = rigids[..., None].invert_apply(points)

    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec**2, dim=-1))

    t_aa_masks = batch["template_all_atom_mask"]
    backbone_mask = t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[..., c]
    backbone_mask_2d = backbone_mask[..., :, None] * backbone_mask[..., None, :]
    if multichain_mask_2d is not None:
        backbone_mask_2d *= multichain_mask_2d

    inv_distance_scalar = inv_distance_scalar * backbone_mask_2d
    unit_vector_data = rigid_vec * inv_distance_scalar[..., None]
    to_concat.extend(torch.unbind(unit_vector_data[..., None, :], dim=-1))
    to_concat.append(backbone_mask_2d[..., None])

    return to_concat


def build_extra_msa_feat(batch: TensorDict) -> torch.Tensor:
    r"""
    Build extra MSA features.

    Requires:
        - extra_msa [N_extra, N_res, 23]
        - extra_msa_has_deletion [N_extra, N_res]
        - extra_msa_deletion_value [N_extra, N_res]

    Returns:
        - extra_msa_feat [N_extra, N_res, 25]
    """

    msa_1hot = one_hot(batch["extra_msa"], 23)
    msa_feat = [
        msa_1hot,
        batch["extra_msa_has_deletion"].unsqueeze(-1),
        batch["extra_msa_deletion_value"].unsqueeze(-1),
    ]
    return torch.cat(msa_feat, dim=-1)
