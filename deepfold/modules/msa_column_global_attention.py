from typing import Optional

import torch
import torch.nn as nn

from deepfold.modules.global_attention import GlobalAttention
from deepfold.modules.layer_norm import LayerNorm


class MSAColumnGlobalAttention(nn.Module):
    """MSA Column Global Attention module.

    Supplementary '1.7.2 Unclustered MSA stack':
    Algorithm 19 MSA global column-wise gated self-attention.

    Args:
        c_e: Extra MSA representation dimension (channels).
        c_hidden: Per-head hidden dimension (channels).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        eps: Epsilon to prevent division by zero.
        chunk_size: Optional chunk size for a batch-like dimension.

    """

    def __init__(
        self,
        c_e: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        eps: float,
        chunk_size: Optional[int],
    ) -> None:
        super().__init__()
        self.layer_norm_m = LayerNorm(c_e)
        self.global_attention = GlobalAttention(
            c_e=c_e,
            c_hidden=c_hidden,
            num_heads=num_heads,
            inf=inf,
            eps=eps,
            chunk_size=chunk_size,
        )

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSA Column Global Attention forward pass.

        Args:
            m: [batch, N_extra_seq, N_res, c_e] extra MSA representation
            mask: [batch, N_extra_seq, N_res] extra MSA mask

        Returns:
            m: [batch, N_extra_seq, N_res, c_e] updated extra MSA representation

        """
        m_transposed = m.transpose(-2, -3)
        # m_transposed: [batch, N_res, N_extra_seq, c_e]

        mask = mask.transpose(-1, -2)
        # mask: [batch, N_res, N_extra_seq]

        m_transposed_normalized = self.layer_norm_m(m_transposed)
        m = self.global_attention(
            m=m_transposed_normalized,
            mask=mask,
            add_transposed_output_to=m,
        )
        # m: [batch, N_extra_seq, N_res, c_e]

        return m
