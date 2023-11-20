# Copyright 2023 DeepFold Team
# Copyright 2021 DeepMind Technologies Limited

from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn

from deepfold.distributed.legacy import col_to_row, row_to_col, scatter
from deepfold.model.alphafold.feats import build_template_angle_feat, build_template_pair_feat
from deepfold.model.alphafold.nn.dropout import DropoutColumnwise, DropoutRowwise
from deepfold.model.alphafold.nn.primitives import Attention, LayerNorm, Linear
from deepfold.model.alphafold.nn.transitions import PairTransition
from deepfold.model.alphafold.nn.triangular_attention import TriangleAttentionEndingNode, TriangleAttentionStartingNode
from deepfold.model.alphafold.nn.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from deepfold.utils.chunk_utils import chunk_layer
from deepfold.utils.tensor_utils import flatten_final_dims, permute_final_dims, tensor_tree_map


class TemplatePointwiseAttention(nn.Module):
    """
    Implements Algorithm 17
    """

    def __init__(self, c_t: int, c_z: int, c_hidden: int, num_heads: int, inf: float) -> None:
        super().__init__()

        self.c_t = c_t
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.inf = inf

        self.mha = Attention(
            self.c_z,
            self.c_t,
            self.c_t,
            self.c_hidden,
            self.c_z,
            self.num_heads,
            gating=False,
        )

    def _chunk(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        mha_inputs = {
            "q_x": z,
            "kv_x": t,
            "biases": biases,
        }
        return chunk_layer(
            self.mha,
            mha_inputs,
            chunk_size=chunk_size,
            num_batch_dims=len(z.shape[:-2]),
        )

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        template_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = 256,  # Small chunk size is harmful for this module
    ) -> torch.Tensor:
        """
        Args:
            t: [*, N_templ, N', N, C_t]
                template embedding
            z: [*, N', N, C_z]
                pair embedding
            template_mask: [*, N_templ]
                template mask
        Returns:
            [*, N, N', C_z] pair embedding update
        """
        if template_mask is None:
            template_mask = t.new_ones(t.shape[-3])

        bias = self.inf * (template_mask[..., None, None, None, None, :] - 1)

        # [*, N', N, 1, C_z]
        z = z.unsqueeze(-2)

        # [*, N', N, N_templ, C_t]
        t = permute_final_dims(t, (1, 2, 0, 3))

        # [*, N', N, 1, C_z]
        biases = [bias]

        if chunk_size is not None and self.training:
            z = self._chunk(z, t, biases, chunk_size)
        else:
            z = self.mha(q_x=z, kv_x=t, biases=biases)

        # [*, N', N, C_z]
        z = z.squeeze(-2)

        return z


class TemplatePairStackBlock(nn.Module):
    def __init__(
        self,
        c_t: int,
        c_hidden_tri_attn: int,
        c_hidden_tri_mul: int,
        num_heads: int,
        pair_transition_n: int,
        dropout_rate: float,
        inf: float,
        **kwargs,
    ) -> None:
        super().__init__()

        self.c_t = c_t
        self.c_hidden_tri_attn = c_hidden_tri_attn
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.num_heads = num_heads
        self.pair_transition_n = pair_transition_n
        self.dropout_rate = dropout_rate
        self.inf = inf

        self.dropout_row = DropoutRowwise(self.dropout_rate)
        self.dropout_col = DropoutColumnwise(self.dropout_rate)

        self.tri_att_start = TriangleAttentionStartingNode(
            self.c_t,
            self.c_hidden_tri_attn,
            self.num_heads,
            inf=inf,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            self.c_t,
            self.c_hidden_tri_attn,
            self.num_heads,
            inf=inf,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            self.c_t,
            self.c_hidden_tri_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            self.c_t,
            self.c_hidden_tri_mul,
        )

        self.pair_transition = PairTransition(
            self.c_t,
            self.pair_transition_n,
        )

    def forward(
        self,
        t: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            t: [*, N_templ, N', N, C_t]
            mask: [*, N, N]
        Returns:
            t: [*, N_templ, N', N, C_t]
        """
        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        mask_row = scatter(mask, -2)  # [*, N', N]
        mask_col = scatter(mask, -1)  # [*, N, N']

        single_templates = [t.unsqueeze(-4) for t in torch.unbind(t, dim=-4)]
        single_template_masks_row = [m.unsqueeze(-3) for m in torch.unbind(mask_row, dim=-3)]
        single_template_masks_col = [m.unsqueeze(-3) for m in torch.unbind(mask_col, dim=-3)]

        trans_masks = single_template_masks_col if _mask_trans else None

        for i in range(len(single_templates)):
            single = single_templates[i]  # [*, 1, N', N, C_t]
            single_mask_row = single_template_masks_row[i]  # [*, 1, N', N]
            single_mask_col = single_template_masks_col[i]  # [*, 1, N, N']
            trans_mask = trans_masks[i]

            # [*, 1, N', N, C_t]
            single = single + self.dropout_row(
                self.tri_att_start(
                    single,
                    mask=single_mask_row,
                    chunk_size=_attn_chunk_size,
                )
            )

            # [*, 1, N, N', C_t]
            single = row_to_col(single)
            single = single + self.dropout_col(
                self.tri_att_end(
                    single,
                    mask=single_mask_col,
                    chunk_size=_attn_chunk_size,
                )
            )

            # [*, 1, N', N, C_t]
            single = col_to_row(single)
            tmu_update = self.tri_mul_out(single, mask=single_mask_row)
            single = single + self.dropout_row(tmu_update)
            del tmu_update

            # [*, 1, N, N', C_t]
            single = row_to_col(single)
            tmu_update = self.tri_mul_in(single, mask=single_mask_col)
            single = single + self.dropout_row(tmu_update)
            del tmu_update

            single = single + self.pair_transition(
                single,
                mask=trans_mask,
                chunk_size=chunk_size,
            )

            # [*, 1, N', N, C_t]
            single = col_to_row(single)

            single_templates[i] = single

        # [*, N_templ, N', N, C_t]
        t = torch.cat(single_templates, dim=-4)

        return t


class TemplatePairStack(nn.Module):
    """
    Implements Algorithm 16.
    """

    def __init__(
        self,
        c_t: int,
        c_hidden_tri_attn: int,
        c_hidden_tri_mul: int,
        num_blocks: int,
        num_heads: int,
        pair_transition_n: int,
        dropout_rate: float,
        inf: float = 1e9,
        **kwargs,
    ):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_hidden_tri_attn:
                Per-head hidden dimension for triangular attention
            c_hidden_tri_attn:
                Hidden dimension for triangular multiplication
            num_blocks:
                Number of blocks in the stack
            pair_transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
            dropout_rate:
                Dropout rate used throughout the stack
        """
        super().__init__()

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = TemplatePairStackBlock(
                c_t=c_t,
                c_hidden_tri_attn=c_hidden_tri_attn,
                c_hidden_tri_mul=c_hidden_tri_mul,
                num_heads=num_heads,
                pair_transition_n=pair_transition_n,
                dropout_rate=dropout_rate,
                inf=inf,
            )
            self.blocks.append(block)

        self.layer_norm = LayerNorm(c_t)

    def forward(
        self,
        t: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            t: [*, N_t, N', N, C_t]
                template embedding
            mask:
                [*, N_t, N', N]
                    mask
        Returns:
            [*, N_t, N', N, C_t] template embedding update
        """

        if mask.shape[-3] == 1:
            expand_idx = list(mask.shape)
            expand_idx[-3] = t.shape[-4]
            mask = mask.expand(*expand_idx)

        blocks = [
            partial(
                b,
                mask=mask,
                chunk_size=chunk_size,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        for b in blocks:
            t = b(t)

        t = self.layer_norm(t)

        return t
