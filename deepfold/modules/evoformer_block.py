from typing import Optional, Tuple

import torch
import torch.nn as nn

import deepfold.core.model_parallel.mappings as cc
import deepfold.core.parallel_state as ps
from deepfold.modules.dropout import DropoutRowwise
from deepfold.modules.evoformer_block_pair_core import EvoformerBlockPairCore
from deepfold.modules.msa_column_attention import MSAColumnAttention
from deepfold.modules.msa_row_attention_with_pair_bias import MSARowAttentionWithPairBias
from deepfold.modules.msa_transition import MSATransition
from deepfold.modules.outer_product_mean import OuterProductMean


class EvoformerBlock(nn.Module):
    """Evoformer Block module.

    Supplementary '1.6 Evoformer blocks': Algorithm 6.
    MSA Transition and Communication are included.

    Args:
        c_m: MSA representation dimension (channels).
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
        eps_opm: Epsilon to prevent division by zero in outer product mean.
        chunk_size_msa_att: Optional chunk size for a batch-like dimension
            in MSA attention.
        chunk_size_opm: Optional chunk size for a batch-like dimension
            in outer product mean.
        chunk_size_tri_att: Optional chunk size for a batch-like dimension
            in triangular attention.

    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_tri_mul: int,
        c_hidden_tri_att: int,
        num_heads_msa: int,
        num_heads_tri: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps_opm: float,
        chunk_size_msa_att: Optional[int],
        chunk_size_opm: Optional[int],
        chunk_size_tri_att: Optional[int],
        outer_product_mean_first: bool = False,
    ) -> None:
        super().__init__()
        self.opm_first = outer_product_mean_first

        self.msa_att_row = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            num_heads=num_heads_msa,
            inf=inf,
            chunk_size=chunk_size_msa_att,
        )
        self.msa_att_col = MSAColumnAttention(
            c_m=c_m,
            c_hidden=c_hidden_msa_att,
            num_heads=num_heads_msa,
            inf=inf,
            chunk_size=chunk_size_msa_att,
        )
        self.msa_dropout_rowwise = DropoutRowwise(
            p=msa_dropout,
        )
        self.msa_transition = MSATransition(
            c_m=c_m,
            n=transition_n,
        )
        self.outer_product_mean = OuterProductMean(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_opm,
            eps=eps_opm,
            chunk_size=chunk_size_opm,
        )
        self.core = EvoformerBlockPairCore(
            c_m=c_m,
            c_z=c_z,
            c_hidden_tri_mul=c_hidden_tri_mul,
            c_hidden_tri_att=c_hidden_tri_att,
            num_heads_tri=num_heads_tri,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
            chunk_size_tri_att=chunk_size_tri_att,
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evoformer Block forward pass.

        Args:
            m: [batch, N_seq, N_res, c_m] MSA representation
            z: [batch, N_res, N_res, c_z] pair representation
            msa_mask: [batch, N_seq, N_res] MSA mask
            pair_mask: [batch, N_res, N_res] pair mask

        Returns:
            m: [batch, N_seq, N_res, c_m] updated MSA representation
            z: [batch, N_res, N_res, c_z] updated pair representation

        """
        if ps.is_enabled():
            msa_mask_row = cc.scatter(msa_mask, dim=-3)
            msa_mask_col = cc.scatter(msa_mask, dim=-2)
            m = self.msa_dropout_rowwise(self.msa_att_row(m=m, z=z, mask=msa_mask_row), add_output_to=m)
            m = cc.row_to_col(m)
            m = self.msa_att_col(m=m, mask=msa_mask_col)

            # TODO: Implement DAP
            raise NotImplementedError("OPM first with DAP is not implemented yet")
        else:
            if self.opm_first:
                z = self.outer_product_mean(m=m, mask=msa_mask, add_output_to=z)
            m = self.msa_dropout_rowwise(self.msa_att_row(m=m, z=z, mask=msa_mask), add_output_to=m)
            m = self.msa_att_col(m=m, mask=msa_mask)

        m = self.msa_transition(m=m, mask=msa_mask)

        if not self.opm_first:
            z = self.outer_product_mean(m=m, mask=msa_mask, add_output_to=z)

        m, z = self.core(
            m=m,
            z=z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
        )

        return m, z
