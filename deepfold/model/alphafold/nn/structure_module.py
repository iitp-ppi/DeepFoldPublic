# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited


import math
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn

from deepfold.common.residue_constants import (
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
)
from deepfold.distributed.legacy import gather
from deepfold.model.alphafold.nn.primitives import LayerNorm, Linear, ipa_point_weights_init_
from deepfold.model.alphafold.utils.feats import frames_and_literature_positions_to_atom14_pos, torsion_angles_to_frames
from deepfold.utils.geometry import Rigid, Rotation
from deepfold.utils.precision import is_fp16_enabled
from deepfold.utils.tensor_utils import dict_multimap, flatten_final_dims, permute_final_dims


class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super().__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, num_blocks, num_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            num_blocks:
                Number of resnet blocks
            num_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_blocks = num_blocks
        self.num_angles = num_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.num_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.num_angles * 2)

        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, num_blocks, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, num_blocks * 2]
        s = self.linear_out(s)

        # [*, num_blocks, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(torch.clamp(torch.sum(s**2, dim=-1, keepdim=True), min=self.eps))
        s = s / norm_denom

        return unnormalized_s, s


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        num_heads: int,
        num_qk_points: int,
        num_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            num_heads:
                Number of attention heads
            num_qk_points:
                Number of query/key points to generate
            num_v_points:
                Number of value points to generate
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.num_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.num_heads * self.num_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.num_heads * (self.num_qk_points + self.num_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        hpv = self.num_heads * self.num_v_points * 3

        self.linear_b = Linear(self.c_z, self.num_heads)

        self.head_weights = nn.Parameter(torch.zeros(num_heads))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.num_heads * (self.c_z + self.c_hidden + self.num_v_points * 4)
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N', C_s] single representation
            z:
                [*, N', N, C_z] pair representation
            r:
                [*, N'] transformation object
            mask:
                [*, N'] mask
        Returns:
            [*, N', C_s] single representation update
        """

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N', H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N', H, C_hidden]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))

        # [*, N', H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.num_heads, -1))

        # [*, N', H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N', H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N', H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N', H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.num_heads, self.num_qk_points, 3))

        # [*, N', H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N', H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N', H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.num_heads, -1, 3))

        # [*, N', H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(kv_pts, [self.num_qk_points, self.num_v_points], dim=-2)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N', N, H]
        b = self.linear_b(z)

        # [*, N', H, C_hidden]
        k_all = gather(k, -3)
        # [*, H, N', N]
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                a = torch.matmul(
                    permute_final_dims(q.float(), (1, 0, 2)),  # [*, H, N, C_hidden]
                    permute_final_dims(k_all.float(), (1, 2, 0)),  # [*, H, C_hidden, N]
                )
        else:
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),  # [*, H, N', C_hidden]
                permute_final_dims(k_all, (1, 2, 0)),  # [*, H, C_hidden, N]
            )

        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        k_pts_all = gather(k_pts, -4)
        # [*, N', N, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts_all.unsqueeze(-5)
        pt_att = pt_att**2

        # [*, N', N, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(*((1,) * len(pt_att.shape[:-2]) + (-1, 1)))
        head_weights = head_weights * math.sqrt(1.0 / (3 * (self.num_qk_points * 9.0 / 2)))
        pt_att = pt_att * head_weights

        # [*, N', N, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        mask_all = gather(mask, -1)
        # [*, N', N]
        square_mask = mask_all.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N', N]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N', H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N', H * C_hidden]
        o = flatten_final_dims(o, 2)

        v_pts_all = gather(v_pts, -4)
        # [*, H, 3, N', P_v]
        o_pt = torch.sum(
            (a[..., None, :, :, None] * permute_final_dims(v_pts_all, (1, 3, 0, 2))[..., None, :, :]),
            dim=-2,
        )

        # [*, N', H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N', H * P_v]
        o_pt_norm = flatten_final_dims(torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2)

        # [*, N', H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, N', H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z.to(dtype=a.dtype))

        # [*, N', H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N', C_s]
        s = self.linear_out(torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1).to(dtype=z[0].dtype))

        return s


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super().__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N, C_s] single representation
        Returns:
            [*, N, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):
    def __init__(self, c, num_layers, dropout_rate):
        super().__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class StructureModule(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_ipa,
        c_resnet,
        num_heads_ipa,
        num_qk_points,
        num_v_points,
        dropout_rate,
        num_blocks,
        num_transition_layers,
        num_resnet_blocks,
        num_angles,
        position_scale,
        epsilon,
        inf,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            num_heads_ipa:
                Number of IPA heads
            num_qk_points:
                Number of query/key points to generate during IPA
            num_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            num_blocks:
                Number of structure module blocks
            num_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            num_resnet_blocks:
                Number of blocks in the angle resnet
            num_angles:
                Number of angles to generate in the angle resnet
            position_scale:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.num_heads_ipa = num_heads_ipa
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_transition_layers = num_transition_layers
        self.num_resnet_blocks = num_resnet_blocks
        self.num_angles = num_angles
        self.position_scale = position_scale
        self.epsilon = epsilon
        self.inf = inf

        # Buffers to be lazily initialized later
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.num_heads_ipa,
            self.num_qk_points,
            self.num_v_points,
            inf=self.inf,
            eps=self.epsilon,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)
        self.transition = StructureModuleTransition(self.c_s, self.num_transition_layers, self.dropout_rate)
        self.bb_update = BackboneUpdate(self.c_s)
        self.angle_resnet = AngleResnet(self.c_s, self.c_resnet, self.num_resnet_blocks, self.num_angles, self.epsilon)

    def forward(self, evoformer_output_dict, aatype, mask=None):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N, C_s] single representation
                    "pair":
                        [*, N, N, C_z] pair representation
            aatype:
                [*, N] amino acid indices
            mask:
                Optional [*, N] sequence mask
        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict["single"]

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid.identity(s.shape[:-1], s.dtype, s.device, self.training, fmt="quat")
        outputs = []
        for _ in range(self.num_blocks):
            # [*, N, C_s]
            s = s + self.ipa(s, z, rigids, mask)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones
            # here
            backb_to_global = Rigid(Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None), rigids.get_trans())

            backb_to_global = backb_to_global.scale_translation(self.position_scale)

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(backb_to_global, angles, aatype)

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )

            scaled_rigids = rigids.scale_translation(self.position_scale)

            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s,
            }

            outputs.append(preds)

            rigids = rigids.stop_rot_gradient()

        del z

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(restype_rigid_group_default_frame, dtype=float_dtype, device=device, requires_grad=False),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(restype_atom14_to_rigid_group, device=device, requires_grad=False),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(restype_atom14_mask, dtype=float_dtype, device=device, requires_grad=False),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions, dtype=float_dtype, device=device, requires_grad=False
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(self, r, f):  # [*, N, 8], [*, N]
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r, f, self.default_frames, self.group_idx, self.atom_mask, self.lit_positions
        )
