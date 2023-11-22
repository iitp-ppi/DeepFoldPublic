# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited

from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from deepfold.distributed.legacy import gather
from deepfold.model.alphafold.nn.primitives import Attention, GlobalAttention, LayerNorm, Linear
from deepfold.utils.chunk_utils import chunk_layer
from deepfold.utils.debug import dump_args
from deepfold.utils.tensor_utils import permute_final_dims


class MSAAttention(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        num_heads: int,
        pair_bias: bool = False,
        c_z: Optional[int] = None,
        inf: float = 1e9,
        eps: float = 1e-10,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            num_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super().__init__()

        self.c_m = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_m)

        self.layer_norm_z = None
        self.linear_z = None

        if self.pair_bias:
            self.layer_norm_z = LayerNorm(self.c_z)
            self.linear_z = Linear(self.c_z, self.num_heads, bias=False, init="normal")

        self.mha = Attention(
            self.c_m,
            self.c_m,
            self.c_m,
            self.c_hidden,
            self.c_m,
            self.num_heads,
            gating=True,
        )

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        biases: Optional[List[torch.Tensor]],
        chunk_size: int,
    ) -> torch.Tensor:
        def fn(m, biases):
            m = self.layer_norm_m(m)
            return self.mha(q_x=m, kv_x=m, biases=biases)

        inputs = {"m": m}
        if biases is not None:
            inputs["biases"] = biases
        else:
            fn = partial(fn, biases=None)

        return chunk_layer(fn, inputs, chunk_size=chunk_size, num_batch_dims=len(m.shape[:-2]))

    def _prep_inputs(
        self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        chunk_size: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_seq, n_res = m.shape[-3:-1]

        if mask is None:
            # [*, S, N]
            mask = m.new_ones(m.shape[:-3] + (n_seq, n_res))

        # [*, S, 1, 1, N]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # TorchScript
        if (self.pair_bias) and (z is not None) and (self.layer_norm_z is not None) and (self.linear_z is not None):
            chunks = []

            for i in range(0, z.shape[-3], chunk_size):
                # [*, I', J, C_z]
                z_chunk = z[..., i : i + chunk_size, :, :]

                # [*, I', J, C_z]
                z_chunk = self.layer_norm_z(z_chunk)

                # [*, I', J, H]
                z_chunk = self.linear_z(z_chunk)

                chunks.append(z_chunk)

            # [*, I', J, H]
            z = torch.cat(chunks, dim=-3)

            # [*, I, J, H]
            z = gather(z, -3)

            # [*, 1, H, I, J]
            z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)

        return m, mask_bias, z

    @dump_args
    def forward(
        self,
        m: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, S', N, C_m] MSA embedding
            z:
                [*, N', N, C_z] pair embedding. Required only if pair_bias is True
            mask:
                [*, S', N] MSA mask
        """

        # m: row [*, S', N, C_m]
        # mask_bias: [*, S', 1, 1, N]
        # z(_bias): [*, 1, H, I, J]
        m, mask_bias, z = self._prep_inputs(m, z, mask)

        biases = [mask_bias]

        if z is not None:
            biases.append(z)

        if chunk_size is not None:
            m = self._chunk(m, biases, chunk_size)
        else:
            m = self.layer_norm_m(m)
            m = self.mha(q_x=m, kv_x=m, biases=biases)

        return m


class MSARowAttentionWithPairBias(MSAAttention):
    """
    Implements Algorithm 7.
    """

    def __init__(self, c_m: int, c_z, c_hidden, num_heads, inf=1e9):
        """
        Args:
            c_m:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Per-head hidden channel dimension
            num_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super().__init__(
            c_m,
            c_hidden,
            num_heads,
            pair_bias=True,
            c_z=c_z,
            inf=inf,
        )


class MSAColumnAttention(nn.Module):
    """
    Implements Algorithm 8.

    By rights, this should also be a subclass of MSAAttention. Alas,
    most inheritance isn't supported by TorchScript.
    """

    def __init__(self, c_m, c_hidden, num_heads, inf=1e9):
        """
        Args:
            c_m:
                MSA channel dimension
            c_hidden:
                Per-head hidden channel dimension
            num_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super().__init__()

        self.c_m = c_m
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.inf = inf

        self._msa_att = MSAAttention(
            c_in=c_m,
            c_hidden=c_hidden,
            num_heads=num_heads,
            pair_bias=False,
            c_z=None,
            inf=inf,
        )

    @dump_args
    def forward(
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
                [*, S, N'] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the
                cost of slower execution. Chunking is not performed by default.
        """

        # [*, N, S, C_in]
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        m = self._msa_att(m, mask=mask, chunk_size=chunk_size)

        # [*, S, N, C_in]
        m = m.transpose(-2, -3)

        return m


class MSAColumnGlobalAttention(nn.Module):
    def __init__(
        self,
        c_e,
        c_hidden,
        num_heads,
        inf=1e9,
        eps=1e-10,
    ):
        super(MSAColumnGlobalAttention, self).__init__()

        self.c_e = c_e
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.inf = inf
        self.eps = eps

        self.layer_norm_m = nn.LayerNorm(c_e)

        self.global_attention = GlobalAttention(
            c_in=self.c_e,
            c_hidden=self.c_hidden,
            c_output=self.c_e,
            num_heads=self.num_heads,
            inf=self.inf,
            eps=self.eps,
        )

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        mha_input = {"m": m, "mask": mask}

        def fn(m, mask):
            m = self.layer_norm_m(m)
            return self.global_attention(m, mask)

        return chunk_layer(fn, mha_input, chunk_size=chunk_size, num_batch_dims=len(m.shape[:-2]))

    @dump_args
    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if mask is None:
            # [*, S, N]
            mask = m.new_ones(m.shape[:-1])

        # [*, N, S, C_m]
        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self.layer_norm_m(m)
            m = self.global_attention(m=m, mask=mask)

        # [*, N, S, C_m]
        m = m.transpose(-2, -3)

        return m
