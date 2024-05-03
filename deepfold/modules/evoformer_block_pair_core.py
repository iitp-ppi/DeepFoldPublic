from typing import Optional, Tuple

import torch
import torch.nn as nn

import deepfold.distributed.model_parallel as mp
from deepfold.modules.dropout import DropoutColumnwise, DropoutRowwise
from deepfold.modules.pair_transition import PairTransition
from deepfold.modules.triangular_attention import TriangleAttentionEndingNode, TriangleAttentionStartingNode
from deepfold.modules.triangular_multiplicative_update import TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing


class EvoformerBlockPairCore(nn.Module):
    """Evoformer Block Pair Core module.

    Pair stack for:
    - Supplementary '1.6 Evoformer blocks': Algorithm 6
    - Supplementary '1.7.2 Unclustered MSA stack': Algorithm 18

    Args:
        c_z: Pair representation dimension (channels).
        c_hidden_msa_att: Hidden dimension in MSA attention.
        c_hidden_opm: Hidden dimension in outer product mean.
        c_hidden_tri_mul: Hidden dimension in multiplicative updates.
        c_hidden_tri_att: Hidden dimension in triangular attention.
        num_heads_msa: Number of heads used in MSA attention.
        num_heads_tri: Number of heads used in triangular attention.
        transition_n: Channel multiplier in transitions.
        msa_dropout: Dropout rate for MSA activations.
        pair_dropout: Dropout rate for pair activations.
        inf: Safe infinity value.
        chunk_size_tri_att: Optional chunk size for a batch-like dimension in triangular attention.

    """

    def __init__(
        self,
        c_z: int,
        c_hidden_tri_mul: int,
        c_hidden_tri_att: int,
        num_heads_tri: int,
        transition_n: int,
        pair_dropout: float,
        inf: float,
        chunk_size_tri_att: Optional[int],
        block_size_tri_mul: Optional[int],
    ) -> None:
        super().__init__()
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z=c_z,
            c_hidden=c_hidden_tri_mul,
            block_size=block_size_tri_mul,
        )
        self.tmo_dropout_rowwise = DropoutRowwise(
            p=pair_dropout,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z=c_z,
            c_hidden=c_hidden_tri_mul,
            block_size=block_size_tri_mul,
        )
        self.tmi_dropout_rowwise = DropoutRowwise(
            p=pair_dropout,
        )
        self.tri_att_start = TriangleAttentionStartingNode(
            c_z=c_z,
            c_hidden=c_hidden_tri_att,
            num_heads=num_heads_tri,
            inf=inf,
            chunk_size=chunk_size_tri_att,
        )
        self.tasn_dropout_rowwise = DropoutRowwise(
            p=pair_dropout,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_z=c_z,
            c_hidden=c_hidden_tri_att,
            num_heads=num_heads_tri,
            inf=inf,
            chunk_size=chunk_size_tri_att,
        )
        self.taen_dropout_columnwise = DropoutColumnwise(
            p=pair_dropout,
        )
        self.pair_transition = PairTransition(
            c_z=c_z,
            n=transition_n,
        )

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Evoformer Block Core forward pass.

        Args:
            z: [batch, N_res, N_res, c_z] pair representation
            pair_mask: [batch, N_res, N_res] pair mask

        Returns:
            z: [batch, N_res, N_res, c_z] updated pair representation

        """
        if mp.is_enabled():
            z = self._forward_dap(z=z, pair_mask=pair_mask)
        else:
            z = self._forward(z=z, pair_mask=pair_mask)
        return z

    def _forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        z = self.tmo_dropout_rowwise(
            self.tri_mul_out(z=z, mask=pair_mask),
            add_output_to=z,
        )
        z = self.tmi_dropout_rowwise(
            self.tri_mul_in(z=z, mask=pair_mask),
            add_output_to=z,
        )
        z = self.tasn_dropout_rowwise(
            self.tri_att_start(z=z, mask=pair_mask),
            add_output_to=z,
        )
        z = self.taen_dropout_columnwise(
            self.tri_att_end(z=z, mask=pair_mask),
            add_output_to=z,
        )
        z = self.pair_transition(z, mask=pair_mask)
        return z

    def _forward_dap(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        pair_mask_row = mp.scatter(pair_mask, dim=-2)
        pair_mask_col = mp.scatter(pair_mask, dim=-1)
        z = self.tmo_dropout_rowwise(
            self.tri_mul_out(z=z, mask=pair_mask_row),
            add_output_to=z,
        )
        z = mp.row_to_col(z)
        z = self.tmi_dropout_rowwise(
            self.tri_mul_in(z=z, mask=pair_mask_col),
            dap_scattered_dim=-2,
            add_output_to=z,
        )
        z = mp.col_to_row(z)
        z = self.tasn_dropout_rowwise(
            self.tri_att_start(z=z, mask=pair_mask_row),
            add_output_to=z,
        )
        z = mp.row_to_col(z)
        z = self.taen_dropout_columnwise(
            self.tri_att_end(z=z, mask=pair_mask_col),
            add_output_to=z,
        )
        z = self.pair_transition(z, mask=pair_mask_col)
        z = mp.col_to_row(z)
        return z
