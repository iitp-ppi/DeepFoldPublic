from typing import Optional

import torch
import torch.nn as nn

import deepfold.distributed as mp
from deepfold.modules.attention import SelfAttentionWithGate
from deepfold.modules.layer_norm import LayerNorm
from deepfold.modules.linear import Linear


class TriangleAttention(nn.Module):
    """Triangle Attention module.

    Supplementary '1.6.6 Triangular self-attention': Algorithms 13 and 14.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).
        num_heads: Number of attention heads.
        ta_type: "starting" or "ending"
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.

    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
        num_heads: int,
        ta_type: str,
        inf: float,
        chunk_size: Optional[int],
    ) -> None:
        super().__init__()
        self._is_starting = {"starting": True, "ending": False}[ta_type]
        self.layer_norm = LayerNorm(c_z)
        self.linear = Linear(c_z, num_heads, bias=False, init="normal")
        self.mha = SelfAttentionWithGate(
            c_qkv=c_z,
            c_hidden=c_hidden,
            num_heads=num_heads,
            inf=inf,
            chunk_size=chunk_size,
        )

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Triangle Attention forward pass.

        Args:
            z: [batch, N_res, N_res, c_z] pair representation
            mask: [batch, N_res, N_res] pair mask

        Returns:
            z_update: [batch, N_res, N_res, c_z] pair representation update

        """
        if not self._is_starting:
            z = z.transpose(-2, -3)
            mask = mask.transpose(-1, -2)
        # z: [batch, N_res, N_res, c_z]
        # mask: [batch, N_res, N_res]

        z = self.layer_norm(z)
        # z: [batch, N_res, N_res, c_z]

        mask = mask.unsqueeze(-2).unsqueeze(-3)
        # mask: [batch, N_res, 1, 1, N_res]

        triangle_bias = self.linear(z)
        # triangle_bias: [batch, N_res, N_res, num_heads]

        if mp.is_enabled():
            triangle_bias = mp.gather(triangle_bias, dim=-3, bwd="all_reduce_sum_split")

        triangle_bias = triangle_bias.movedim(-1, -3).unsqueeze(-4).contiguous()
        # triangle_bias: [batch, 1, num_heads, N_res, N_res]

        z = self.mha(
            input_qkv=z,
            mask=mask,
            bias=triangle_bias,
        )
        # z: [batch, N_res, N_res, c_z]

        if not self._is_starting:
            z = z.transpose(-2, -3)
        # z: [batch, N_res, N_res, c_z]

        return z


class TriangleAttentionStartingNode(TriangleAttention):
    """Triangle Attention Starting Node module.

    Supplementary '1.6.6 Triangular self-attention':
    Algorithm 13 Triangular gated self-attention around starting node.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.

    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        chunk_size: Optional[int],
    ) -> None:
        super().__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            num_heads=num_heads,
            ta_type="starting",
            inf=inf,
            chunk_size=chunk_size,
        )


class TriangleAttentionEndingNode(TriangleAttention):
    """Triangle Attention Ending Node module.

    Supplementary '1.6.6 Triangular self-attention':
    Algorithm 14 Triangular gated self-attention around ending node.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.

    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        chunk_size: Optional[int],
    ) -> None:
        super().__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            num_heads=num_heads,
            ta_type="ending",
            inf=inf,
            chunk_size=chunk_size,
        )
