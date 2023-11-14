# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited

from typing import Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from deepfold.common import residue_constants as rc
from deepfold.utils.geometry import Rigid, Rigid3Array
from deepfold.utils.tensor_utils import batched_gather

FeatureDict = Mapping[str, torch.Tensor]


def pseudo_beta_fn(
    aatype: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Construct pseudo beta-carbon position array and mask optionally.

    Args:
        aatype:
            One-hot amino acid type tensor
        all_atom_position:
            All atom position tensor
        all_atom_mask:
            All atom mask
    Return:
        Tensor of pseudo beta-carbon position (and mask).

    Note:
        For gylcine-residues (G, GLY) alpha-carbon is considered as beta-carbon.
    """

    is_gly = aatype == rc.restype_order["G"]
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]

    pseudo_beta = torch.where(
        is_gly[..., None].expand(*((-1,) * len(is_gly.shape)), 3),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_mask[..., ca_idx],
            all_atom_mask[..., cb_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


def atom14_to_atom37(atom14_data: torch.Tensor, batch: FeatureDict) -> torch.Tensor:
    """
    Convert atom14 to atom37 representation.
    """

    atom37_data = batched_gather(
        atom14_data,
        batch["residx_atom37_to_atom14"],
        dim=-2,
        num_batch_dims=len(
            atom14_data.shape[:-2],
        ),
    )
    atom37_data = atom37_data * batch["atom37_atom_exists"][..., None]

    return atom37_data


def build_template_angle_feat(template_feats: FeatureDict) -> FeatureDict:
    template_aatype = template_feats["template_aatype"]
    torsion_angles_sin_cos = template_feats["template_torsion_angles_sin_cos"]
    alt_torsion_angles_sin_cos = template_feats["template_alt_torsion_angles_sin_cos"]
    torsion_angles_mask = template_feats["template_torsion_angles_mask"]
    template_angle_feat = torch.cat(
        [
            nn.functional.one_hot(template_aatype, 22),
            torsion_angles_sin_cos.reshape(*torsion_angles_sin_cos.shape[:-2], 14),
            alt_torsion_angles_sin_cos.reshape(*alt_torsion_angles_sin_cos.shape[:-2], 14),
            torsion_angles_mask,
        ],
        dim=-1,
    )

    return template_angle_feat


def dgram_from_positions(
    pos: torch.Tensor,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    num_bins: float = 39,
    inf: float = 1e8,
) -> torch.Tensor:
    """
    Build a distogram from positions
    """

    dgram = torch.sum((pos[..., None, :] - pos[..., None, :, :]) ** 2, dim=-1, keepdim=True)
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)

    return dgram


def build_template_pair_feat(
    batch: FeatureDict,
    min_bin,
    max_bin,
    num_bins,
    use_unit_vector=False,
    eps=1e-20,
    inf=1e08,
) -> torch.Tensor:
    """
    Build template pair features.
    """

    template_mask = batch["template_pseudo_beta_mask"]
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

    # Compute distogram (this seems to differ slightly from Alg. 5)
    tpb = batch["template_pseudo_beta"]
    dgram = dgram_from_positions(tpb, min_bin, max_bin, num_bins, inf)

    to_concat = [dgram, template_mask_2d[..., None]]

    aatype_one_hot = nn.functional.one_hot(
        batch["template_aatype"],
        rc.restype_num + 2,
    )

    n_res = batch["template_aatype"].shape[-1]
    to_concat.append(aatype_one_hot[..., None, :, :].expand(*aatype_one_hot.shape[:-2], n_res, -1, -1))
    to_concat.append(aatype_one_hot[..., None, :].expand(*aatype_one_hot.shape[:-2], -1, n_res, -1))

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
    template_mask = t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[..., c]
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

    inv_distance_scalar = inv_distance_scalar * template_mask_2d
    unit_vector = rigid_vec * inv_distance_scalar[..., None]

    if not use_unit_vector:
        unit_vector = unit_vector * 0.0

    to_concat.extend(torch.unbind(unit_vector[..., None, :], dim=-1))
    to_concat.append(template_mask_2d[..., None])

    act = torch.cat(to_concat, dim=-1)
    act = act * template_mask_2d[..., None]

    return act


def build_extra_msa_feat(batch: FeatureDict) -> torch.Tensor:
    msa_1hot = nn.functional.one_hot(batch["extra_msa"], 23)
    msa_feat = [
        msa_1hot,
        batch["extra_has_deletion"].unsqueeze(-1),
        batch["extra_deletion_value"].unsqueeze(-1),
    ]
    return torch.cat(msa_feat, dim=-1)


def torsion_angles_to_frames(
    r: Union[Rigid, Rigid3Array],
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    default_frames: torch.Tensor,
) -> Rigid:
    """
    Compute rigid group frames from torsion angles.

    Alg. 24:2-10
    Alg. 25
    """

    rigid_type = type(r)

    # [*, N, 8, 4, 4]
    default_4x4 = default_frames[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = rigid_type.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.shape + (4, 4))
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:3] = alpha

    all_rots = rigid_type.from_tensor_4x4(all_rots)
    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = rigid_type.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    r: Union[Rigid, Rigid3Array],
    aatype: torch.Tensor,
    default_frames: torch.Tensor,
    group_idx: torch.Tensor,
    atom_mask: torch.Tensor,
    lit_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Put atom literature positions (atom14 encoding) in each rigid group.

    Alg. 24:11

    Args:
        r:
            frames
        aatyle:
            one-hot encoded atom types
    Returns:
        Tensor of predicted atom positions
    """

    # [*, N, 14, 4, 4]
    # default_4x4 = default_frames[aatype, ...]

    # [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 14]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions
