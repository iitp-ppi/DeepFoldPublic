from typing import Optional

import torch
import torch.nn as nn

from deepfold.model.v2.modules.attention import SelfAttentionWithGate
from deepfold.model.v2.modules.layer_norm import LayerNorm


class MSAColumnAttention(nn.Module):
    """MSA Column Attention module.

    Supplementary '1.6.2 MSA column-wise gated self-attention': Algorithm 8.

    Args:
        c_m: MSA representation dimension (channels).
        c_hidden: Per-head hidden dimension (channels).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.

    """

    def __init__(
        self,
        c_m: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        chunk_size: Optional[int],
    ) -> None:
        super(MSAColumnAttention, self).__init__()
        self.layer_norm_m = LayerNorm(c_m)
        self.mha = SelfAttentionWithGate(
            c_qkv=c_m,
            c_hidden=c_hidden,
            num_heads=num_heads,
            inf=inf,
            chunk_size=chunk_size,
        )

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSA Column Attention forward pass.

        Args:
            m: [batch, N_seq, N_res, c_m] MSA representation
            mask: [batch, N_seq, N_res] MSA mask

        Returns:
            m: [batch, N_seq, N_res, c_m] updated MSA representation

        """
        m_transposed = m.transpose(-2, -3)
        # m_transposed: [batch, N_res, N_seq, c_m]

        mask = mask.transpose(-1, -2)
        # mask: [batch, N_res, N_seq]

        mask = mask.unsqueeze(-2).unsqueeze(-3)
        # mask: [batch, N_res, 1, 1, N_seq]

        m_transposed_normalized = self.layer_norm_m(m_transposed)
        m = self.mha(
            input_qkv=m_transposed_normalized,
            mask=mask,
            bias=None,
            add_transposed_output_to=m,
        )
        # m: [batch, N_seq, N_res, c_m]

        return m
