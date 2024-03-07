from typing import Optional

import torch
import torch.nn as nn

import deepfold.core.model_parallel.mappings as cc
import deepfold.core.parallel_state as ps
from deepfold.model.v2.modules.attention import SelfAttentionWithGate
from deepfold.model.v2.modules.layer_norm import LayerNorm
from deepfold.model.v2.modules.linear import Linear


class MSARowAttentionWithPairBias(nn.Module):
    """MSA Row Attention With Pair Bias module.

    Supplementary '1.6.1 MSA row-wise gated self-attention with pair bias': Algorithm 7.

    Args:
        c_m: MSA (or Extra MSA) representation dimension (channels).
        c_z: Pair representation dimension (channels).
        c_hidden: Per-head hidden dimension (channels).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.

    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        chunk_size: Optional[int],
    ) -> None:
        super(MSARowAttentionWithPairBias, self).__init__()
        self.layer_norm_m = LayerNorm(c_m)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, num_heads, bias=False, init="normal")
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
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSA Row Attention With Pair Bias forward pass.

        Args:
            m: [batch, N_seq, N_res, c_m] MSA (or Extra MSA) representation
            z: [batch, N_res, N_res, c_z] pair representation
            mask: [batch, N_seq, N_res] MSA (or Extra MSA) mask

        Returns:
            m_update: [batch, N_seq, N_res, c_m]
                MSA (or Extra MSA) representation update

        """
        mask = mask.unsqueeze(-2).unsqueeze(-3)
        # mask: [batch, N_seq, 1, 1, N_res]

        z = self.layer_norm_z(z)
        z = self.linear_z(z)
        if ps.is_enabled():
            z = cc.gather_from_model_parallel_region(z, dim=-3, bwd="all_reduce_sum_split")
        # z: [batch, N_res, N_res, num_heads]

        z = z.movedim(-1, -3).unsqueeze(-4)
        # z: [batch, 1, num_heads, N_res, N_res]

        m = self.layer_norm_m(m)
        m = self.mha(
            input_qkv=m,
            mask=mask,
            bias=z,
        )
        # m: [batch, N_seq, N_res, c_m]

        return m


# TODO: Chunk
