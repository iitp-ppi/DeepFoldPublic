from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn

from deepfold.distributed.legacy import gather
from deepfold.model.alphafold.nn.primitives import Attention, LayerNorm, Linear
from deepfold.utils.chunk_utils import chunk_layer
from deepfold.utils.debug import dump_args
from deepfold.utils.tensor_utils import permute_final_dims


class TriangleAttention(nn.Module):
    def __init__(
        self,
        c_z: int,
        c_hidden: int,
        num_heads: int,
        starting: bool = True,
        inf: float = 1e9,
    ) -> None:
        super().__init__()

        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_z)
        self.linear = Linear(c_z, self.num_heads, bias=False, init="normal")
        self.mha = Attention(self.c_z, self.c_z, self.c_z, self.c_hidden, self.c_z, self.num_heads, gating=True)

    @torch.jit.ignore
    def _chunk(self, x: torch.Tensor, biases: List[torch.Tensor], chunk_size: int) -> torch.Tensor:
        mha_inputs = {"q_x": x, "kv_x": x, "biases": biases}
        return chunk_layer(
            partial(self.mha),
            mha_inputs,
            chunk_size=chunk_size,
            num_batch_dims=len(x.shape[:-2]),
        )

    @dump_args
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [*, N', N, C_z]
            mask: [*, N', N]
        Returns:
            [*, N', N, C_z]
        """
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I', J, C_z] / [*, J', I, C_z]
        x = self.layer_norm(x)

        # [*, I', 1, 1, J] / [*, J', 1, 1, I]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, I', J, H] / [*, J', I, H]
        triangle_bias = self.linear(x)

        # [*, I, J, H] / [*, J, I, H]
        triangle_bias = gather(triangle_bias, -3)

        # [*, H, I, J] -> [*, 1, H, I, J]
        triangle_bias = permute_final_dims(triangle_bias, (2, 0, 1)).unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            x = self._chunk(x, biases, chunk_size)
        else:
            x = self.mha(q_x=x, kv_x=x, biases=biases)

        if not self.starting:
            x = x.transpose(-2, -3)

        # [*, I', J, C_z]
        return x


class TriangleAttentionStartingNode(TriangleAttention):
    """
    Implements Algorithm 13.
    """

    def __init__(self, c_z: int, c_hidden: int, num_heads: int, inf: float = 1e9) -> None:
        super().__init__(c_z, c_hidden, num_heads, starting=True, inf=inf)


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """

    def __init__(self, c_z: int, c_hidden: int, num_heads: int, inf: float = 1e9) -> None:
        super().__init__(c_z, c_hidden, num_heads, starting=False, inf=inf)
