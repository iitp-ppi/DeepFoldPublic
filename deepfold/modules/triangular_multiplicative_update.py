from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.distributed as mp
import deepfold.modules.inductor as inductor
from deepfold.modules.layer_norm import LayerNorm
from deepfold.modules.linear import Linear
from deepfold.utils.iter_utils import slice_generator
from deepfold.utils.precision import is_fp16_enabled


class TriangleMultiplicativeUpdate(nn.Module):
    """Triangle Multiplicative Update module.

    Supplementary '1.6.5 Triangular multiplicative update': Algorithms 11 and 12.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).
        tmu_type: "outgoing" or "incoming"

    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
        tmu_type: str,
        block_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._is_outgoing = {"outgoing": True, "incoming": False}[tmu_type]
        self.block_size = block_size

        self.linear_ab_p = Linear(c_z, c_hidden * 2, init="default")
        self.linear_ab_g = Linear(c_z, c_hidden * 2, init="gating")
        # self.linear_a_p = Linear(c_z, c_hidden, bias=True, init="default")
        # self.linear_a_g = Linear(c_z, c_hidden, bias=True, init="gating")
        # self.linear_b_p = Linear(c_z, c_hidden, bias=True, init="default")
        # self.linear_b_g = Linear(c_z, c_hidden, bias=True, init="gating")
        self.linear_g = Linear(c_z, c_z, bias=True, init="gating")
        self.linear_z = Linear(c_hidden, c_z, bias=True, init="final")
        self.layer_norm_in = LayerNorm(c_z)
        self.layer_norm_out = LayerNorm(c_hidden)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Triangle Multiplicative Update forward pass.

        Args:
            z: [batch, N_res, N_res, c_z] pair representation
            mask: [batch, N_res, N_res] pair mask

        Returns:
            z_update: [batch, N_res, N_res, c_z] pair representation update

        """
        z = self.layer_norm_in(z)
        # z: [batch, N_res, N_res, c_z]

        mask = mask.unsqueeze(-1)
        # mask: [batch, N_res, N_res, 1]

        if not self.training and self.block_size is not None:
            return self._chunk_2d(z, mask)

        # TODO: Fusion with a.float, b.float (?)
        a, b = _compute_projections(
            z,
            mask,
            self.linear_ab_g.weight,
            self.linear_ab_g.bias,
            self.linear_ab_p.weight,
            self.linear_ab_p.bias,
        )  # .chunk(2, dim=-1)

        if mp.is_enabled():
            if self._is_outgoing:
                b = mp.gather(b, dim=-3, bwd="all_reduce_sum_split")
            else:
                a = mp.gather(a, dim=-2, bwd="all_reduce_sum_split")

        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)
        # x: [batch, N_res, N_res, c_hidden]

        del a, b

        x = self.layer_norm_out(x)
        # x: [batch, N_res, N_res, c_hidden]

        x = _compute_output(
            x,
            z,
            self.linear_z.weight,
            self.linear_z.bias,
            self.linear_g.weight,
            self.linear_g.bias,
        )
        # x: [batch, N_res, N_res, c_z]

        return x

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:

        if self._is_outgoing:
            a = a.movedim(-1, -3)
            b = b.swapdims(-1, -3)
        else:
            a = a.swapdims(-1, -3)
            b = b.movedim(-1, -3)

        p = torch.matmul(a, b)

        return p.movedim(-3, -1)

    def _chunk_2d(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z: [B, N', N, C]
            mask: [B, N', N, 1]

        Returns:
            z: [*, N', N, C]

        Notes:
            Avoid too small block size (<256).
        """
        out = torch.empty_like(z)

        par_dim = z.shape[-3] if self._is_outgoing else z.shape[-2]  # N'

        for i_begin, i_end in slice_generator(0, par_dim, self.block_size):
            if self._is_outgoing:
                z_i = z[:, i_begin:i_end, :, :]  # [B, N', N, C]
                a_chunk, _ = _compute_projections(
                    z_i,
                    mask[:, i_begin:i_end, :, :],
                    self.linear_ab_g.weight,
                    self.linear_ab_g.bias,
                    self.linear_ab_p.weight,
                    self.linear_ab_p.bias,
                )  # a_chunk: [B, I', K, C], b_chunk: [B, J', K, C]
                a_chunk = a_chunk.movedim(-1, -3)  # [B, C, I', K]
            else:  # is_incoming
                z_i = z[:, :, i_begin:i_end, :]  # [B, N, N', C]
                _, b_chunk = _compute_projections(
                    z_i,
                    mask[:, :, i_begin:i_end, :],
                    self.linear_ab_g.weight,
                    self.linear_ab_g.bias,
                    self.linear_ab_p.weight,
                    self.linear_ab_p.bias,
                )  # a_chunk: [B, K, I', C], b_chunk: [B, K, J', C]
                b_chunk = b_chunk.movedim(-1, -3)  # [B, C, K, J']

                for j_begin, j_end in slice_generator(0, par_dim, self.block_size):
                    if self._is_outgoing:
                        z_j = z[:j_begin:j_end, :, :]
                        _, b_chunk = _compute_projections(
                            z_j,
                            mask[:, j_begin:j_end, :, :],
                            self.linear_ab_g.weight,
                            self.linear_ab_g.bias,
                            self.linear_ab_p.weight,
                            self.linear_ab_p.bias,
                        )  # a_chunk: [B, K, I', C], b_chunk: [B, K, J', C]
                        b_chunk = b_chunk.swapdims(-1, -3)  # [B, C, K, J']
                    else:
                        z_j = z[:, :, j_begin:j_end, :]
                        a_chunk, _ = _compute_projections(
                            z_j,
                            mask[:, :, j_begin:j_end, :],
                            self.linear_ab_g.weight,
                            self.linear_ab_g.bias,
                            self.linear_ab_p.weight,
                            self.linear_ab_p.bias,
                        )  # a_chunk: [B, K, I', C], b_chunk: [B, K, J', C]
                        a_chunk = a_chunk.swapdims(-1, -3)  # [B, C, I', K]

                        if mp.is_enabled():
                            for r in range(mp.size()):
                                if self._is_outgoing:
                                    if r == mp.rank():
                                        buf = b_chunk.clone()
                                    else:
                                        buf = torch.empty_like(b_chunk)
                                    buf = mp.broadcast(buf, r)
                                    x_chunk = torch.matmul(a_chunk, buf)
                                    del buf
                                else:
                                    if r == mp.rank():
                                        buf = a_chunk.clone()
                                    else:
                                        buf = torch.empty_like(a_chunk)
                                    buf = mp.broadcast(buf, r)
                                    x_chunk = torch.matmul(buf, b_chunk)
                                    del buf
                                x_chunk = x_chunk.movedim(-3, -1)
                                j_global_begin = par_dim * r + j_begin
                                j_global_end = min(j_global_begin + self.block_size, par_dim * (r + 1))

                                if self._is_outgoing:
                                    out[:, i_begin:i_end, j_global_begin:j_global_end, :] = x_chunk
                                else:
                                    out[:, j_global_begin:j_global_end, i_begin:i_end, :] = x_chunk
                                del x_chunk
                        else:
                            x_chunk = torch.matmul(a_chunk, b_chunk).movedim(-3, -1)
                            if self._is_outgoing:
                                out[:, i_begin:i_end, j_begin:j_end, :] = x_chunk
                            else:
                                out[:, j_begin:j_end, i_begin:i_end, :] = x_chunk

        for i_begin, i_end in slice_generator(0, z.shape[-3], self.block_size):
            for j_begin, j_end in slice_generator(0, z.shape[-2], self.block_size):
                z_chunk = z[:, i_begin:i_end, j_begin:j_end, :]
                x_chunk = out[:, i_begin:i_end, j_begin:j_end, :]
                x_chunk = self.layer_norm_out(x_chunk)
                x_chunk = _compute_output(
                    x_chunk,
                    z_chunk,
                    self.linear_z.weight,
                    self.linear_z.bias,
                    self.linear_g.weight,
                    self.linear_g.bias,
                )
                out[:, i_begin:i_end, j_begin:j_end, :] = x_chunk

        return out


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """Triangle Multiplication Outgoing module.

    Supplementary '1.6.5 Triangular multiplicative update':
    Algorithm 11 Triangular multiplicative update using "outgoing" edges.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).

    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
        block_size: Optional[int],
    ) -> None:
        super().__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            tmu_type="outgoing",
            block_size=block_size,
        )


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """Triangle Multiplication Incoming module.

    Supplementary '1.6.5 Triangular multiplicative update':
    Algorithm 12 Triangular multiplicative update using "incoming" edges.

    Args:
        c_z: Pair or template representation dimension (channels).
        c_hidden: Hidden dimension (channels).

    """

    def __init__(
        self,
        c_z: int,
        c_hidden: int,
        block_size: Optional[int],
    ) -> None:
        super().__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            tmu_type="incoming",
            block_size=block_size,
        )


def _compute_projections_eager(
    z: torch.Tensor,
    mask: torch.Tensor,
    w_ab_g: torch.Tensor,
    b_ab_g: torch.Tensor,
    w_ab_p: torch.Tensor,
    b_ab_p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ab = F.linear(z, w_ab_g, b_ab_g)
    ab = torch.sigmoid(ab) * mask
    ab = ab * F.linear(z, w_ab_p, b_ab_p)
    return ab


_compute_projections_jit = torch.compile(_compute_projections_eager)


def _compute_projections(
    z: torch.Tensor,
    mask: torch.Tensor,
    w_ab_g: torch.Tensor,
    b_ab_g: torch.Tensor,
    w_ab_p: torch.Tensor,
    b_ab_p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if inductor.is_enabled():
        compute_projections_fn = _compute_projections_jit
    else:
        compute_projections_fn = _compute_projections_eager
    return compute_projections_fn(z, mask, w_ab_g, b_ab_g, w_ab_p, b_ab_p).chunk(2, dim=-1)


def _compute_output_eager(
    x: torch.Tensor,
    z: torch.Tensor,
    w_z: torch.Tensor,
    b_z: torch.Tensor,
    w_g: torch.Tensor,
    b_g: torch.Tensor,
) -> torch.Tensor:
    x = F.linear(x, w_z, b_z)
    g = torch.sigmoid(F.linear(z, w_g, b_g))
    x = x * g
    return x


_compute_output_jit = torch.compile(_compute_output_eager)


def _compute_output(
    x: torch.Tensor,
    z: torch.Tensor,
    w_z: torch.Tensor,
    b_z: torch.Tensor,
    w_g: torch.Tensor,
    b_g: torch.Tensor,
) -> torch.Tensor:
    if inductor.is_enabled():
        compute_output_fn = _compute_output_jit
    elif inductor.is_enabled_and_autograd_off():
        compute_output_fn = _compute_output_jit
    else:
        compute_output_fn = _compute_output_eager
    return compute_output_fn(x, z, w_z, b_z, w_g, b_g)
