# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited

from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn

from deepfold.distributed.legacy import col_to_row, row_to_col, scatter
from deepfold.model.alphafold.dist_layers import EvoformerScatter
from deepfold.model.alphafold.nn.dropout import DropoutColumnwise, DropoutRowwise
from deepfold.model.alphafold.nn.msa import MSAColumnAttention, MSAColumnGlobalAttention, MSARowAttentionWithPairBias
from deepfold.model.alphafold.nn.outer_product_mean import ParallelOuterProductMean
from deepfold.model.alphafold.nn.primitives import Linear
from deepfold.model.alphafold.nn.transitions import MSATransition, PairTransition
from deepfold.model.alphafold.nn.triangular_attention import TriangleAttentionEndingNode, TriangleAttentionStartingNode
from deepfold.model.alphafold.nn.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from deepfold.utils.debug import dump_args


class MSACore(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_attn: int,
        num_heads_msa: int,
        transition_n: int,
        msa_dropout: float,
        inf: float,
        eps: float,
    ) -> None:
        super().__init__()

        self.msa_att_row = MSARowAttentionWithPairBias(c_m, c_z, c_hidden_msa_attn, num_heads_msa, inf=inf)

        self.msa_att_col = MSAColumnAttention(c_m, c_hidden_msa_attn, num_heads_msa, inf=inf)

        self.msa_dropout_layer = DropoutRowwise(msa_dropout)

        self.msa_transition = MSATransition(c_m, transition_n)

    @dump_args
    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m: [*, S, N', C_m]
            z: [*, N', N, C_z]
        """

        msa_mask_row = scatter(msa_mask, -2)
        msa_mask_col = scatter(msa_mask, -1)

        msa_trans_mask = msa_mask_col if _mask_trans else None

        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        # If not opm_first
        m = col_to_row(m)  # [*, S', N, C_m]

        m = m + self.msa_dropout_layer(self.msa_att_row(m, z, mask=msa_mask_row, chunk_size=_attn_chunk_size))

        m = row_to_col(m)  # [*, S, N', C_m]

        m = m + self.msa_att_col(m, mask=msa_mask_col, chunk_size=_attn_chunk_size)

        m = m + self.msa_transition(m, mask=msa_trans_mask, chunk_size=chunk_size)

        return m  # [*, S, N', C_m]


class PairCore(nn.Module):
    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_attn: int,
        num_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        inf: float,
        eps: float,
    ) -> None:
        super().__init__()

        # TODO: fuse_projection_weights

        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_hidden_mul)
        self.tri_att_start = TriangleAttentionStartingNode(c_z, c_hidden_pair_attn, num_heads_pair, inf=inf)
        self.tri_att_end = TriangleAttentionEndingNode(c_z, c_hidden_pair_attn, num_heads_pair, inf=inf)

        self.pair_transition = PairTransition(c_z, transition_n)

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)
        self.ps_dropout_col_layer = DropoutColumnwise(pair_dropout)

    @dump_args
    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            z: [*, N', N, C_z]
            pair_mask: [*, N, N]

        """

        pair_mask_row = scatter(pair_mask, -2)
        pair_mask_col = scatter(pair_mask, -1)

        # DeepMind doesn't mask these transitions in the source code.
        # Therefore, `_mask_trans` should be `False` to better approximate the exact activations.
        trans_mask = pair_mask_col if _mask_trans else None

        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        tmu_update = self.tri_mul_out(z, mask=pair_mask_row, chunk_size=chunk_size)
        z = z + self.ps_dropout_row_layer(tmu_update)
        del tmu_update

        z = row_to_col(z)
        tmu_update = self.tri_mul_in(z, mask=pair_mask_col, chunk_size=chunk_size)
        z = z + self.ps_dropout_row_layer(tmu_update)
        del tmu_update

        z = col_to_row(z)
        z = z + self.ps_dropout_row_layer(self.tri_att_start(z, mask=pair_mask_row, chunk_size=_attn_chunk_size))

        z = row_to_col(z)
        z = z + self.ps_dropout_col_layer(self.tri_att_end(z, mask=pair_mask_col, chunk_size=_attn_chunk_size))

        z = z + self.pair_transition(z, mask=trans_mask, chunk_size=chunk_size)

        z = col_to_row(z)

        return z  # [*, I', J, C_z]


class EvoformerBlock(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_attn: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_attn: int,
        num_heads_msa: int,
        num_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
    ):
        super().__init__()

        self.msa = MSACore(
            c_m,
            c_z,
            c_hidden_msa_attn,
            num_heads_msa,
            transition_n,
            msa_dropout,
            inf,
            eps,
        )

        self.communication = ParallelOuterProductMean(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_opm,
            eps=1e-3,  # DO NOT MODIFY!
        )

        self.pair = PairCore(
            c_z,
            c_hidden_mul,
            c_hidden_pair_attn,
            num_heads_pair,
            transition_n,
            pair_dropout,
            inf,
            eps,
        )

    @dump_args
    def forward(
        self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m: [*, S, N', C_m]
            z: [*, N', N, C_z]
            msa_mask: [*, S, N]
            pair_mask: [*, N, N]
        """

        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        input_tensors = [m, z]
        m, z = input_tensors

        # TODO: opm_first

        # [*, S, N', C_m]
        m = self.msa(m, z, msa_mask, chunk_size=chunk_size, _mask_trans=_mask_trans, _attn_chunk_size=_attn_chunk_size)

        # If not opm_first
        z = z + self.communication(m, mask=msa_mask, chunk_size=chunk_size)

        # [*, N', N, C_z]
        z = self.pair(z, pair_mask, chunk_size=chunk_size, _mask_trans=_mask_trans, _attn_chunk_size=_attn_chunk_size)

        return m, z


class EvoformerStack(nn.Module):
    """
    Main Evoformer trunk.

    Implements Algorithm 6.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_attn: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_attn: int,
        c_s: int,
        num_heads_msa: int,
        num_heads_pair: int,
        num_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        clear_cache_between_blocks: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair channel dimension
            c_hidden_msa_attn:
                Hidden dimension in MSA attention
            c_hidden_opm:
                Hidden dimension in outer product mean module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_attn:
                Hidden dimension in triangular attention
            c_s:
                Channel dimension of the output "single" embedding
            num_heads_msa:
                Number of heads used for MSA attention
            num_heads_pair:
                Number of heads used for pair attention
            num_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
            msa_dropout:
                Dropout rate for MSA activations
            pair_dropout:
                Dropout used for pair activations
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        super().__init__()

        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.scatter_msa_features = EvoformerScatter()

        self.blocks = nn.ModuleList()

        for _ in range(num_blocks):
            block = EvoformerBlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_msa_attn=c_hidden_msa_attn,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_attn=c_hidden_pair_attn,
                num_heads_msa=num_heads_msa,
                num_heads_pair=num_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps,
            )
            self.blocks.append(block)

        self.linear = Linear(c_m, c_s)

    @dump_args
    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                [*, S, N', C_m] MSA embedding
            z:
                [*, N', N, C_z] pair embedding
            msa_mask:
                [*, S, N] MSA mask
            pair_mask:
                [*, N, N] pair mask
            chunk_size:
                Inference-time subbatch size.
        Returns:
            m:
                [*, S, N', C_m] MSA embedding
            z:
                [*, N', N, C_z] pair embedding
            s:
                [*, N', C_s] single embedding (or None if extra MSA stack)
        """

        m, msa_mask = self.scatter_msa_features(m, msa_mask)

        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if self.clear_cache_between_blocks:

            def block_with_cache_clear(block, *args):
                torch.cuda.empty_cache()
                return block(*args)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        for b in blocks:
            m, z = b(m, z)

        s = self.linear(m[..., 0, :, :])

        return m, z, s


class ExtraMSABlock(EvoformerBlock):
    def __init__(
        self,
        c_e: int,
        c_z: int,
        c_hidden_msa_attn: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_attn: int,
        num_heads_msa: int,
        num_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
    ) -> None:
        super().__init__(
            c_e,
            c_z,
            c_hidden_msa_attn,
            c_hidden_opm,
            c_hidden_mul,
            c_hidden_pair_attn,
            num_heads_msa,
            num_heads_pair,
            transition_n,
            msa_dropout,
            pair_dropout,
            inf,
            eps,
        )

        self.msa.msa_att_col = MSAColumnGlobalAttention(
            c_e=c_e,
            c_hidden=c_hidden_msa_attn,
            num_heads=num_heads_msa,
            inf=inf,
            eps=eps,
        )

    @dump_args
    def forward(
        self,
        e: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            e: [*, S_e, N', C_e]
            z: [*, N', N, C_z]
        """

        e, z = super().forward(
            e,
            z,
            msa_mask,
            pair_mask,
            chunk_size=chunk_size,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size,
        )

        return e, z


class ExtraMSAStack(nn.Module):
    """
    Implement Algorithm 18.
    """

    def __init__(
        self,
        c_e: int,
        c_z: int,
        c_hidden_msa_attn: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_attn: int,
        num_heads_msa: int,
        num_heads_pair: int,
        num_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        clear_cache_between_blocks: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList()

        for _ in range(num_blocks):
            block = ExtraMSABlock(
                c_e=c_e,
                c_z=c_z,
                c_hidden_msa_attn=c_hidden_msa_attn,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_attn=c_hidden_pair_attn,
                num_heads_msa=num_heads_msa,
                num_heads_pair=num_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps,
            )
            self.blocks.append(block)

    @dump_args
    def forward(
        self,
        e: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                _mask_trans=_mask_trans,
                _attn_chunk_size=_attn_chunk_size,
            )
            for b in self.blocks
        ]

        if self.clear_cache_between_blocks:

            def block_with_cache_clear(block, **kwargs):
                torch.cuda.empty_cache()
                return block(**kwargs)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        for b in blocks:
            e, z = b(e, z)

        return z
