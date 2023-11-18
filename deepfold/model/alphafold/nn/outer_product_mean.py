# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited


from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from deepfold.distributed.legacy import gather, scatter
from deepfold.model.alphafold.nn.primitives import Linear
from deepfold.utils.chunk_utils import chunk_layer
from deepfold.utils.debug import dump_args
from deepfold.utils.precision import is_fp16_enabled
from deepfold.utils.tensor_utils import flatten_final_dims


class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden**2, c_z, init="final")

    def _opm(self, a, b):
        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        return outer

    @torch.jit.ignore
    def _chunk(self, a: torch.Tensor, b: torch.Tensor, chunk_size: int) -> torch.Tensor:
        # Since that batch_dim in this case is not a true batch dimension,
        # we need to iterate over it ourselves.
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size=chunk_size,
                num_batch_dims=1,
            )
            out.append(outer)

        if len(out) == 1:
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)
        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

    def _forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, N_res, C_m]
        ln = self.layer_norm(m)

        # [*, N_seq, N_res, C]
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln)
        a = a * mask

        b = self.linear_2(ln)
        b = b * mask

        del ln

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        if chunk_size is None:
            outer = self._opm(a, b)
        else:
            outer = self._chunk(a, b, chunk_size)

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        # [*, N_res, N_res, C_z]
        outer = outer / norm

        return outer

    @dump_args
    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(m.float(), mask, chunk_size)
        else:
            return self._forward(m, mask, chunk_size)


class ParallelOuterProductMean(nn.Module):
    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden**2, c_z, init="final")

    def _opm(self, a: torch.Tensor, b: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: [*, N', S, C]
            b: [*, N, S, C]
            norm: [*, N', N, 1]
        Returns:
            [*, N', N, C_z]
        """
        # [*, N', N, C, C]
        outer = torch.einsum("...isc,...jsd->...ijcd", a, b)

        # [*, N', N, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N', N, C_z]
        outer = self.linear_out(outer)

        # [*, N', N, C_z]
        outer = outer / norm

        return outer

    def _chunk(
        self,
        a: torch.Tensor,
        b_all: torch.Tensor,
        norm: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """
        Args:
            a: [*, N', S, C]
            b_all: [*, N, S, C]
            norm: [*, N', N, 1]
        Returns:
            [*, N', N, C_z]
        """
        outer_shape = (*norm.shape[:-1], self.linear_out.out_features)

        # [*, N', N, C_z]
        out = a.new_zeros(outer_shape)  # TODO: Zero?
        a_reshape = a.view((-1,) + a.shape[-3:])
        b_reshape = b_all.view((-1,) + b_all.shape[-3:])
        norm_reshape = norm.view((-1,) + norm.shape[-3:])

        for a_prime, b_prime, norm_prime in zip(a_reshape, b_reshape, norm_reshape):
            chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime, "norm": norm_prime},
                chunk_size=chunk_size,
                num_batch_dims=1,
                _out=out,
            )

        return out

    def _forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, S, N', C_m] MSA embedding
            mask:
                [*, S, N] MSA mask
        Returns:
            [*, N', N, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, S, N']
        msa_mask_col = scatter(mask, -1)

        # [*, S, N', C_m]
        ln = self.layer_norm(m)

        # [*, S, N', C]
        a = self.linear_1(ln)
        a = a * msa_mask_col.unsqueeze(-1)

        # [*, S, N', C]
        b = self.linear_2(ln)
        b = b * msa_mask_col.unsqueeze(-1)
        # [*, S, N, C]
        b_all = gather(b, -2)

        del ln

        # [*, N', S, C]
        a = a.transpose(-2, -3)
        # [*, N, S, C]
        b_all = b_all.transpose(-2, -3)

        # [*, N', N, 1]
        norm = torch.einsum("...si,...sj->...ij", msa_mask_col, mask).unsqueeze(-1)
        norm = norm + self.eps

        if chunk_size is not None:
            outer = self._chunk(a, b_all, norm, chunk_size)
        else:
            outer = self._opm(a, b_all, norm)

        return outer

    @dump_args
    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(m.float(), mask, chunk_size)
        else:
            return self._forward(m, mask, chunk_size)
