from typing import Optional

import torch
import torch.nn as nn

from deepfold.distributed.legacy import (
    broadcast_async_begin,
    broadcast_async_end,
    broadcast_sync,
    gather,
    gather_async_begin,
    gather_async_end,
    get_rank,
    get_world_size,
)
from deepfold.model.alphafold.nn.primitives import LayerNorm, Linear
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

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
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

        if chunk_size is None:
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

        else:
            if self._outgoing:
                x = self._chunk_outgoing(z, mask, chunk_size)
            else:
                x = self._chunk_incoming(z, mask, chunk_size)

        return x  # [*, I, J', C] / [*, I', J, C]

    def _chunk_outgoing(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        world_size = get_world_size()
        rank = get_rank()

        para_dim = z.shape[-3]
        # z: [B, I', J, C] for outgoing
        out = torch.empty_like(z)

        # mask: [B, I', J, 1]

        # Left projection; i: row (dim=-3)
        for i in range(0, para_dim, chunk_size):
            z_i = z[..., i : i + chunk_size, :, :]
            z_i = self.layer_norm_in(z_i)

            # TODO: Fused

            g = self.sigmoid(self.linear_a_g(z_i))
            a = self.linear_a_p(z_i)
            a = a * g
            del g

            # [*, I', J, C]
            a = mask[..., i : i + chunk_size, :, :] * a

            # Right projection; j: row (dim=-3)
            for j in range(0, para_dim, chunk_size):
                z_j = z[..., j : j + chunk_size, :, :]
                z_j = self.layer_norm_in(z_j)

                # TODO: Fused

                g = self.sigmoid(self.linear_b_g(z_j))
                b = self.linear_b_p(z_j)
                b = b * g
                del g

                # [*, I', J, C]
                b = mask[..., j : j + chunk_size, :, :] * b

                work = None
                b_buf = torch.empty_like(b)

                for k in range(0, world_size):
                    if world_size > 1:
                        if work:
                            # If `work` is not `None` then collect last broadcast
                            broadcast_async_end(work)
                            if k != rank:
                                b_recv = b_buf.clone()
                        else:
                            # Initialize first broadcast
                            if k == rank:
                                broadcast_sync(k, b, host=True)
                            else:
                                b_buf = broadcast_sync(k, b, host=False)
                                b_recv = b_buf.clone()

                        # Launch next broadcast
                        if (k + 1) != world_size:
                            if k + 1 == rank:
                                work = broadcast_async_begin(k + 1, b, host=True)
                            else:
                                work = broadcast_async_begin(k + 1, b_buf, host=False)

                    if k == rank:  # Broadcast it's own right projection
                        p = torch.einsum("...ikc,...jkc->...ijc", a, b)

                    else:  # Receive others broadcast
                        p = torch.einsum("...ikc,...jkc->...ijc", a, b_recv)

                    j_0 = para_dim * k + j
                    j_1 = min(j_0 + chunk_size, para_dim * (k + 1))

                    out[..., i : i + chunk_size, j_0:j_1, :] = p

        for i in range(0, para_dim, chunk_size):
            z_0 = z[..., i : i + chunk_size, :, :]
            g = self.sigmoid(self.linear_g(self.layer_norm_in(z_0)))

            x = self.linear_z(self.layer_norm_out(out[..., i : i + chunk_size, :, :]))
            x = x * g
            del g

            z[..., i : i + chunk_size, :, :] = z_0 + x

        return z

    def _chunk_incoming(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        world_size = get_world_size()
        rank = get_rank()

        para_dim = z.shape[-2]
        # z: [B, I, J', C] for outgoing
        out = torch.empty_like(z)

        # mask: [B, I, J', 1]

        # Right projection; i: col (dim=-2)
        for i in range(0, para_dim, chunk_size):
            z_i = z[..., :, i : i + chunk_size, :]
            z_i = self.layer_norm_in(z_i)

            # TODO: Fused

            g = self.sigmoid(self.linear_b_g(z_i))
            b = self.linear_b_p(z_i)
            b = b * g
            del g

            # [*, I, J', C]
            b = mask[..., :, i : i + chunk_size, :] * b

            # Left projection; j: col (dim=-2)
            for j in range(0, para_dim, chunk_size):
                z_j = z[..., :, j : j + chunk_size, :]
                z_j = self.layer_norm_in(z_j)

                # TODO: Fused

                g = self.sigmoid(self.linear_a_g(z_j))
                a = self.linear_a_p(z_j)
                a = a * g
                del g

                # [*, I, J', C]
                a = mask[..., :, j : j + chunk_size, :] * a

                work = None
                a_buf = torch.empty_like(b)

                for k in range(0, world_size):
                    if world_size > 1:
                        if work:
                            # If `work` is not `None` then collect last broadcast
                            broadcast_async_end(work)
                            if k != rank:
                                a_recv = a_buf.clone()
                        else:
                            # Initialize first broadcast
                            if k == rank:
                                broadcast_sync(k, a, host=True)
                            else:
                                a_buf = broadcast_sync(k, a, host=False)
                                a_recv = a_buf.clone()

                        # Launch next broadcast
                        if (k + 1) != world_size:
                            if k + 1 == rank:
                                work = broadcast_async_begin(k + 1, a, host=True)
                            else:
                                work = broadcast_async_begin(k + 1, a_buf, host=False)

                    if k == rank:  # Broadcast it's own right projection
                        p = torch.einsum("...kic,...kjc->...ijc", a, b)

                    else:  # Receive others broadcast
                        p = torch.einsum("...kic,...kjc->...ijc", a_recv, b)

                    j_0 = para_dim * k + j
                    j_1 = min(j_0 + chunk_size, para_dim * (k + 1))

                    out[..., j_0:j_1, i : i + chunk_size, :] = p

        for i in range(0, para_dim, chunk_size):
            z_0 = z[..., :, i : i + chunk_size, :]
            g = self.sigmoid(self.linear_g(self.layer_norm_in(z_0)))

            x = self.linear_z(self.layer_norm_out(out[..., :, i : i + chunk_size, :]))
            x = x * g
            del g

            z[..., :, i : i + chunk_size, :] = z_0 + x

        return z


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
