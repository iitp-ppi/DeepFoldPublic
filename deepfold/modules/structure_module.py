from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.common.residue_constants as rc
from deepfold.modules.angle_resnet import AngleResnet
from deepfold.modules.backbone_update import BackboneUpdate
from deepfold.modules.invariant_point_attention import InvariantPointAttention, InvariantPointAttentionMultimer
from deepfold.modules.layer_norm import LayerNorm
from deepfold.modules.linear import Linear
from deepfold.modules.single_transition import SingleTransition
from deepfold.utils.geometry import Rigid3Array
from deepfold.utils.geometry.quat_rigid import QuatRigid
from deepfold.utils.rigid_utils import Rigid, Rotation
from deepfold.utils.tensor_utils import add


class StructureModule(nn.Module):
    """Structure Module.

    Supplementary '1.8 Structure module': Algorithm 20.

    Args:
        c_s: Single representation dimension (channels).
        c_z: Pair representation dimension (channels).
        c_hidden_ipa: Hidden dimension in invariant point attention.
        c_hidden_ang_res: Hidden dimension in angle resnet.
        num_heads_ipa: Number of heads used in invariant point attention.
        num_qk_points: Number of query/key points in invariant point attention.
        num_v_points: Number of value points in invariant point attention.
        dropout_rate: Dropout rate in structure module.
        num_blocks: Number of shared blocks in the forward pass.
        num_ang_res_blocks: Number of blocks in angle resnet.
        num_angles: Number of angles in angle resnet.
        scale_factor: Scale translation factor.
        inf: Safe infinity value.
        eps: Epsilon to prevent division by zero.

    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_ipa: int,
        c_hidden_ang_res: int,
        num_heads_ipa: int,
        num_qk_points: int,
        num_v_points: int,
        is_multimer: bool,
        dropout_rate: float,
        num_blocks: int,
        num_ang_res_blocks: int,
        num_angles: int,
        scale_factor: float,
        inf: float,
        eps: float,
    ) -> None:
        super(StructureModule, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden_ipa = c_hidden_ipa
        self.c_hidden_ang_res = c_hidden_ang_res
        self.num_heads_ipa = num_heads_ipa
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_ang_res_blocks = num_ang_res_blocks
        self.num_angles = num_angles
        self.scale_factor = scale_factor
        self.is_multimer = is_multimer  # is_multimer
        self.inf = inf
        self.eps = eps

        self.layer_norm_s = LayerNorm(c_s)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_in = Linear(c_s, c_s, bias=True, init="default")
        if not self.is_multimer:
            self.ipa = InvariantPointAttention(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden_ipa,
                num_heads=num_heads_ipa,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                separate_kv=self.is_multimer,  # is_multimer
                inf=inf,
                eps=eps,
            )
        else:
            self.ipa = InvariantPointAttentionMultimer(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden_ipa,
                num_heads=num_heads_ipa,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                inf=inf,
                eps=eps,
            )

        self.ipa_dropout = nn.Dropout(dropout_rate)
        self.layer_norm_ipa = LayerNorm(c_s)
        self.transition = SingleTransition(
            c_s=c_s,
            dropout_rate=dropout_rate,
        )
        if self.is_multimer:  # is_multimer
            self.bb_update = QuatRigid(c_hidden=c_s, full_quat=False)
        else:
            self.bb_update = BackboneUpdate(c_s=c_s)
        self.angle_resnet = AngleResnet(
            c_s=c_s,
            c_hidden=c_hidden_ang_res,
            num_blocks=num_ang_res_blocks,
            num_angles=num_angles,
            eps=eps,
        )
        # buffers:
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        aatype: torch.Tensor,
        inplace_safe: bool,
    ) -> Dict[str, torch.Tensor]:
        if self.is_multimer:  # is_multimer
            outputs = self._forward_multimer(s, z, mask, aatype, inplace_safe)
        else:
            outputs = self._forward_monomer(s, z, mask, aatype, inplace_safe)
        return outputs

    def _forward_monomer(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        aatype: torch.Tensor,
        inplace_safe: bool,
    ) -> Dict[str, torch.Tensor]:
        """Structure Module forward pass.

        Args:
            s: [batch, N_res, c_s] single representation
            z: [batch, N_res, N_res, c_z] pair representation
            mask: [batch, N_res] sequence mask
            aatype: [batch, N_res] amino acid indices

        Returns:
            dictionary containing:
                "sm_frames": [batch, num_blocks, N_res, 7]
                "sm_sidechain_frames": [batch, num_blocks, N_res, 8, 4, 4]
                "sm_unnormalized_angles": [batch, num_blocks, N_res, 7, 2]
                "sm_angles": [batch, num_blocks, N_res, 7, 2]
                "sm_positions": [batch, num_blocks, N_res, 14, 3]
                "sm_states": [batch, num_blocks, N_res, c_s]
                "sm_single": [batch, N_res, c_s] updated single representation

        """
        self._initialize_buffers(dtype=s.dtype, device=s.device)

        s = self.layer_norm_s(s)
        # s: [batch, N_res, c_s]

        z = self.layer_norm_z(z)
        # z: [batch, N_res, N_res, c_z]

        s_initial = s
        # s_initial: [batch, N_res, c_s]

        s = self.linear_in(s)
        # s: [batch, N_res, c_s]

        rigids = Rigid.identity(
            shape=s.shape[:-1],
            dtype=s.dtype,
            device=s.device,
            requires_grad=self.training,
            fmt="quat",
        )

        outputs = []

        for _ in range(self.num_blocks):
            s = add(s, self.ipa(s=s, z=z, r=rigids, mask=mask, inplace_safe=inplace_safe), inplace_safe)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            # s: [batch, N_res, c_s]

            s = self.transition(s, inplace_safe)
            # s: [batch, N_res, c_s]

            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones here
            backb_to_global = Rigid(
                rots=Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                trans=rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(self.scale_factor)

            unnormalized_angles, angles = self.angle_resnet(s=s, s_initial=s_initial)
            # unnormalized_angles: [batch, N_res, num_angles, 2]
            # angles: [batch, N_res, num_angles, 2]

            all_frames_to_global = _torsion_angles_to_frames(
                r=backb_to_global,
                alpha=angles,
                aatype=aatype,
                rrgdf=self.default_frames,
            )

            pred_xyz = _frames_and_literature_positions_to_atom14_pos(
                r=all_frames_to_global,
                aatype=aatype,
                default_frames=self.default_frames,
                group_idx=self.group_idx,
                atom_mask=self.atom_mask,
                lit_positions=self.lit_positions,
            )
            # pred_xyz: [batch, N_res, 14, 3]

            scaled_rigids = rigids.scale_translation(self.scale_factor)

            output = {
                "sm_frames": scaled_rigids.to_tensor_7(),
                "sm_sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "sm_unnormalized_angles": unnormalized_angles,
                "sm_angles": angles,
                "sm_positions": pred_xyz,
                "sm_states": s,
            }

            outputs.append(output)

            rigids = rigids.stop_rot_gradient()

        outputs = self._stack_tensors(outputs, dim=1)

        outputs["sm_single"] = s

        return outputs

    def _initialize_buffers(self, dtype: torch.dtype, device: torch.device) -> None:
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    rc.restype_rigid_group_default_frame,
                    dtype=dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    rc.restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    rc.restype_atom14_mask,
                    dtype=dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    rc.restype_atom14_rigid_group_positions,
                    dtype=dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def _stack_tensors(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        dim: int,
    ) -> Dict[str, torch.Tensor]:
        stacked = {}
        for key in list(outputs[0].keys()):
            stacked[key] = torch.stack(
                tensors=[output[key] for output in outputs],
                dim=dim,
            )
        return stacked

    def _forward_multimer(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        aatype: torch.Tensor,
        inplace_safe: bool,
    ):
        self._initialize_buffers(dtype=s.dtype, device=s.device)

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(z)

        # [*, N, C_s]
        s_initial = s

        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid3Array.identity(s.shape[:-1], s.device)

        outputs = []

        for _ in range(self.num_blocks):
            # [*, N, C_s]
            s = add(s, self.ipa(s=s, z=z, r=rigids, mask=mask, inplace_safe=inplace_safe), inplace_safe)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)

            s = self.transition(s, inplace_safe)

            # [*, N]
            rigids = rigids @ self.bb_update(s)

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = _torsion_angles_to_frames(
                r=rigids.scale_translation(self.scale_factor),
                alpha=angles,
                aatype=aatype,
                rrgdf=self.default_frames,
            )

            pred_xyz = _frames_and_literature_positions_to_atom14_pos(
                r=all_frames_to_global,
                aatype=aatype,
                default_frames=self.default_frames,
                group_idx=self.group_idx,
                atom_mask=self.atom_mask,
                lit_positions=self.lit_positions,
            )

            output = {
                "sm_frames": rigids.scale_translation(self.scale_factor).to_tensor(),
                "sm_sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "sm_unnormalized_angles": unnormalized_angles,
                "sm_angles": angles,
                "sm_positions": pred_xyz,
                "sm_states": s,
            }

            outputs.append(output)

            rigids = rigids.stop_rot_gradient()

        outputs = self._stack_tensors(outputs, dim=1)
        outputs["sm_single"] = s

        return outputs


def _torsion_angles_to_frames(
    r: Rigid | Rigid3Array,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
) -> Rigid | Rigid3Array:
    rigid_type = type(r)

    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

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
    # This follows the original code rather than the supplement,
    # which uses different indices.

    # all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots = alpha.new_zeros(default_r.shape + (4, 4))
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:3] = alpha

    # all_rots = Rigid(Rotation(rot_mats=all_rots), None)
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

    # all_frames_to_global = r[..., None].compose_jit(all_frames_to_bb)
    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def _frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames: torch.Tensor,
    group_idx: torch.Tensor,
    atom_mask: torch.Tensor,
    lit_positions: torch.Tensor,
) -> torch.Tensor:
    # [*, N, 14, 4, 4]
    # default_4x4 = default_frames[aatype, ...]

    # [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask = F.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions
