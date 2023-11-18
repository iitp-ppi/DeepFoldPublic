from typing import Optional

import torch
import torch.nn as nn

from deepfold.distributed.legacy import (
    broadcast_async_begin,
    broadcast_async_end,
    broadcast_sync,
    gather,
    get_rank,
    get_world_size,
)
from deepfold.model.alphafold.nn.primitives import LayerNorm, Linear
from deepfold.utils.debug import dump_args
from deepfold.utils.precision import is_fp16_enabled
from deepfold.utils.tensor_utils import permute_final_dims


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implement Algorithms 11 and 12.
    """

    def __init__(self, c_z: int, c_hidden: int, _outgoing: bool = True) -> None:
        super().__init__()

        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        # chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            a:
                Left projection
            b:
                Right projection
        """
        #    incoming      outgoing
        # a [*, K, I', C] [*, I', K, C]
        # b [*, K, J', C] [*, J', K, C]

        # TODO: Async

        if not self._outgoing:  # incoming
            # [*, C, I', K]
            a = permute_final_dims(a, (2, 1, 0))
            # [*, C, I, K]
            a = gather(a.contiguous(), -2)
            # [*, C, K, J']
            b = permute_final_dims(b, (2, 0, 1))
        else:  # outgoing
            # [*, J, K, C]
            b = gather(b.contiguous(), -3)
            # [*, C, I', K]
            a = permute_final_dims(a, (2, 0, 1))
            # [*, C, K, J]
            b = permute_final_dims(b, (2, 1, 0))

        # [*, C, I, J'] / [*, C, I', J]
        p = torch.matmul(a, b)
        # [*, I, J', C] / [*, I', J, C]
        return permute_final_dims(p, (1, 2, 0))

    @dump_args
    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size=None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J', C_z] for incoming
                [*, I', J, C_z] for outgoing
            mask:
                [*, I, J'] for incoming
                [*, I', J] for outgoing
        Returns:
            [*, I, J', C_z] for incoming
            [*, I', J, C_z] for outgoing
        """

        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)  # [*, I, J', 1] / [*, I', J, 1]

        z = self.layer_norm_in(z)

        # TODO: Fused

        a = mask
        # [*, I, J', C] / [*, I', J, C]
        a = a * self.sigmoid(self.linear_a_g(z))
        a = a * self.linear_a_p(z)

        b = mask
        # [*, I, J', C] / [*, I', J, C]
        b = b * self.sigmoid(self.linear_b_g(z))
        b = b * self.linear_b_p(z)

        if is_fp16_enabled():
            # Prevent overflow of matmul
            a_std = a.std()
            b_std = b.std()
            if a_std != 0.0 and b_std != 0.0:
                a = a / a_std
                b = b / b_std

        # [*, I, J', C] / [*, I', J, C]
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)

        del a, b

        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x  # [*, I, J', C] / [*, I', J, C]


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """

    def __init__(self, c_z: int, c_hidden: int) -> None:
        super().__init__(c_z, c_hidden, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """

    def __init__(self, c_z: int, c_hidden: int) -> None:
        super().__init__(c_z, c_hidden, _outgoing=False)
