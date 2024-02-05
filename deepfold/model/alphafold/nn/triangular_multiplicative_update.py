# Copyright 2023 DeapFold Team


from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from deepfold.distributed.legacy import broadcast, gather, get_rank, get_world_size
from deepfold.model.alphafold.nn.primitives import LayerNorm, Linear
from deepfold.utils.chunk_utils import chunk_layer
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

            x = self.linear_z(self.layer_norm_out(x))
            x = x * self.sigmoid(self.linear_g(z))
        else:
            if self._outgoing:
                # x = self._chunk_outgoing(z, mask, chunk_size)
                x = chunk_layer(
                    partial(self._chunk_outgoing, chunk_size=chunk_size),
                    {"z": z, "mask": mask},
                    chunk_size=chunk_size,
                    num_batch_dims=len(z.shape[:-3]),
                )
            else:
                # x = self._chunk_incoming(z, mask, chunk_size)
                x = chunk_layer(
                    partial(self._chunk_incoming, chunk_size=chunk_size),
                    {"z": z, "mask": mask},
                    chunk_size=chunk_size,
                    num_batch_dims=len(z.shape[:-3]),
                )

        return x  # [*, I, J', C] / [*, I', J, C]

    def _chunk_outgoing(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """
        Args:
            z: [*, N', N, C]
            mask: [*, N', N, 1]
        Returns:
            [*, N', N, C]
        """
        world_size = get_world_size()
        rank = get_rank()
        out = torch.empty_like(z)

        para_dim = z.shape[-3]  # N'

        for i in range(0, para_dim, chunk_size):
            i_end = min(para_dim, i + chunk_size)

            z_i = z[:, i:i_end, :, :]
            z_i = self.layer_norm_in(z_i)

            a = mask[:, i:i_end, :, :]
            # [I'', J, C]
            a = a * self.sigmoid(self.linear_a_g(z_i))
            a = a * self.linear_a_p(z_i)
            # [C, I'', K]
            a = permute_final_dims(a, (2, 0, 1))

            for j in range(0, para_dim, chunk_size):
                j_end = min(para_dim, j + chunk_size)

                z_j = z[:, j:j_end, :, :]
                z_j = self.layer_norm_in(z_j)

                b = mask[:, j:j_end, :, :]
                # [I'', J, C]
                b = b * self.sigmoid(self.linear_b_g(z_j))
                b = b * self.linear_b_p(z_j)
                # [C, K, J'']
                b = permute_final_dims(b, (2, 1, 0))
                b = b.contiguous()

                for r in range(0, world_size):
                    if world_size > 1:
                        if r == rank:
                            b_buf = b.clone()
                        else:
                            b_buf = torch.empty_like(b)

                        b_buf = broadcast(b_buf, r)
                        p = torch.matmul(a, b_buf)
                    else:
                        p = torch.matmul(a, b)  # [C, I'', J'']

                    p = permute_final_dims(p, (1, 2, 0))  # [I'', J'', C]
                    j_global = para_dim * r + j
                    out[:, i:i_end, j_global : min(j_global + chunk_size, para_dim * (r + 1)), :] = p

        for i in range(0, z.shape[-3], chunk_size):
            i_end = min(z.shape[-3], i + chunk_size)
            for j in range(0, z.shape[-2], chunk_size):
                j_end = min(z.shape[-2], j + chunk_size)

                z_chunk = z[:, i:i_end, j:j_end, :]
                g_chunk = self.sigmoid(self.linear_g(self.layer_norm_in(z_chunk)))
                x_chunk = out[:, i:i_end, j:j_end, :]
                x_chunk = self.linear_z(self.layer_norm_out(x_chunk))
                out[:, i:i_end, j:j_end, :] = x_chunk * g_chunk

        return out

    def _chunk_incoming(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """
        Args:
            z: [*, N, N', C]
            mask: [*, N, N', 1]
        Returns:
            [*, N, N', C]
        """
        world_size = get_world_size()
        rank = get_rank()
        out = torch.empty_like(z)

        para_dim = z.shape[-2]  # N'

        for i in range(0, para_dim, chunk_size):
            i_end = min(para_dim, i + chunk_size)

            z_i = z[:, :, i:i_end, :]
            z_i = self.layer_norm_in(z_i)

            b = mask[:, :, i:i_end, :]
            # [K, J'', C]
            b = b * self.sigmoid(self.linear_b_g(z_i))
            b = b * self.linear_b_p(z_i)
            # [C, J'', K]
            b = permute_final_dims(b, (2, 0, 1))

            for j in range(0, para_dim, chunk_size):
                j_end = min(para_dim, j + chunk_size)

                z_j = z[:, :, j:j_end, :]
                z_j = self.layer_norm_in(z_j)

                a = mask[:, :, j:j_end, :]
                # [I'', K, C]
                a = a * self.sigmoid(self.linear_a_g(z_j))
                a = a * self.linear_a_p(z_j)
                # [C, K, I'']
                a = permute_final_dims(a, (2, 1, 0))
                a = a.contiguous()

                for r in range(0, world_size):
                    if world_size > 1:
                        if r == rank:
                            a_buf = a.clone()
                        else:
                            a_buf = torch.empty_like(a)

                        a_buf = broadcast(a_buf, r)
                        p = torch.matmul(a_buf, b)
                    else:
                        p = torch.matmul(a, b)  # [C, I'', J'']

                    p = permute_final_dims(p, (1, 2, 0))  # [I'', J'', C]
                    j_global = para_dim * r + j
                    out[:, j_global : min(j_global + chunk_size, para_dim * (r + 1)), i:i_end, :] = p

        for i in range(0, z.shape[-3], chunk_size):
            i_end = min(z.shape[-3], i + chunk_size)
            for j in range(0, z.shape[-2], chunk_size):
                j_end = min(z.shape[-2], j + chunk_size)

                z_chunk = z[:, i:i_end, j:j_end, :]
                g_chunk = self.sigmoid(self.linear_g(self.layer_norm_in(z_chunk)))
                x_chunk = out[:, i:i_end, j:j_end, :]
                x_chunk = self.linear_z(self.layer_norm_out(x_chunk))
                out[:, i:i_end, j:j_end, :] = x_chunk * g_chunk

        return out


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
