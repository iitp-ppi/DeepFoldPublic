from typing import Optional

import torch
import torch.nn as nn

import deepfold.core.model_parallel.mappings as cc
import deepfold.core.parallel_state as ps
from deepfold.model.v2.modules.dropout import DropoutColumnwise, DropoutRowwise
from deepfold.model.v2.modules.pair_transition import PairTransition
from deepfold.model.v2.modules.triangular_attention import TriangleAttentionEndingNode, TriangleAttentionStartingNode
from deepfold.model.v2.modules.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class TemplatePairBlock(nn.Module):
    """Template Pair Block module.

    Supplementary '1.7.1 Template stack': Algorithm 16.

    Args:
        c_t: Template representation dimension (channels).
        c_hidden_tri_att: Hidden dimension in triangular attention.
        c_hidden_tri_mul: Hidden dimension in multiplicative updates.
        num_heads_tri: Number of heads used in triangular attention.
        pair_transition_n: Channel multiplier in pair transition.
        dropout_rate: Dropout rate for pair activations.
        inf: Safe infinity value.
        chunk_size_tri_att: Optional chunk size for a batch-like dimension
            in triangular attention.

    """

    def __init__(
        self,
        c_t: int,
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        num_heads_tri: int,
        pair_transition_n: int,
        dropout_rate: float,
        inf: float,
        chunk_size_tri_att: Optional[int],
    ) -> None:
        super().__init__()
        self.tri_att_start = TriangleAttentionStartingNode(
            c_z=c_t,
            c_hidden=c_hidden_tri_att,
            num_heads=num_heads_tri,
            inf=inf,
            chunk_size=chunk_size_tri_att,
        )
        self.tasn_dropout_rowwise = DropoutRowwise(
            p=dropout_rate,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_z=c_t,
            c_hidden=c_hidden_tri_att,
            num_heads=num_heads_tri,
            inf=inf,
            chunk_size=chunk_size_tri_att,
        )
        self.taen_dropout_columnwise = DropoutColumnwise(
            p=dropout_rate,
        )
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z=c_t,
            c_hidden=c_hidden_tri_mul,
        )
        self.tmo_dropout_rowwise = DropoutRowwise(
            p=dropout_rate,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z=c_t,
            c_hidden=c_hidden_tri_mul,
        )
        self.tmi_dropout_rowwise = DropoutRowwise(
            p=dropout_rate,
        )
        self.pair_transition = PairTransition(
            c_z=c_t,
            n=pair_transition_n,
        )

    def forward(
        self,
        t: torch.tensor,
        mask: torch.tensor,
    ) -> torch.Tensor:
        """Template Pair Block forward pass.

        Args:
            t: [batch, N_templ, N_res, N_res, c_t] template representation
            mask: [batch, N_res, N_res] pair mask

        Returns:
            t: [batch, N_templ, N_res, N_res, c_t] updated template representation

        """
        t_list = list(torch.unbind(t, dim=-4))

        N_templ = len(t_list)

        for i in range(N_templ):
            t = t_list[i]
            # t: [batch, N_res, N_res, c_t]

            if ps.is_enabled():
                mask_row = cc.scatter(mask, dim=-3)
                mask_col = cc.scatter(mask, dim=-2)
                t = self.tasn_dropout_rowwise(
                    self.tri_att_start(z=t, mask=mask_row),
                    add_output_to=t,
                )
                t = cc.row_to_col(t)
                t = self.taen_dropout_columnwise(
                    self.tri_att_end(z=t, mask=mask_col),
                    add_output_to=t,
                )
                t = cc.col_to_row(t)
                t = self.tmo_dropout_rowwise(
                    self.tri_mul_out(z=t, mask=mask_row),
                    add_output_to=t,
                )
                t = cc.row_to_col(t)
                t = self.tmi_dropout_rowwise(
                    self.tri_mul_in(z=t, mask=mask_col),
                    dap_scattered_dim=2,
                    add_output_to=t,
                )
                t = self.pair_transition(z=t, mask=mask_col)
                t = cc.col_to_row(t)
            else:
                t = self.tasn_dropout_rowwise(
                    self.tri_att_start(z=t, mask=mask),
                    add_output_to=t,
                )
                t = self.taen_dropout_columnwise(
                    self.tri_att_end(z=t, mask=mask),
                    add_output_to=t,
                )
                t = self.tmo_dropout_rowwise(
                    self.tri_mul_out(z=t, mask=mask),
                    add_output_to=t,
                )
                t = self.tmi_dropout_rowwise(
                    self.tri_mul_in(z=t, mask=mask),
                    add_output_to=t,
                )
                t = self.pair_transition(z=t, mask=mask)

            t_list[i] = t

        t = torch.stack(t_list, dim=-4)
        # t: [batch, N_templ, N_res, N_res, c_t]

        return t
