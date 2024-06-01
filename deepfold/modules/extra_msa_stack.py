from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpointing_fn

import deepfold.distributed.model_parallel as mp
from deepfold.modules.extra_msa_block import ExtraMSABlock
from deepfold.utils.dist_utils import get_pad_size, pad_tensor


class ExtraMSAStack(nn.Module):
    """Extra MSA Stack module.

    Supplementary '1.7.2 Unclustered MSA stack'.

    Args:
        c_e: Extra MSA representation dimension (channels).
        c_z: Pair representation dimension (channels).
        c_hidden_msa_att: Hidden dimension in MSA attention.
        c_hidden_opm: Hidden dimension in outer product mean.
        c_hidden_tri_mul: Hidden dimension in multiplicative updates.
        c_hidden_tri_att: Hidden dimension in triangular attention.
        num_heads_msa: Number of heads used in MSA attention.
        num_heads_tri: Number of heads used in triangular attention.
        num_blocks: Number of blocks in the stack.
        transition_n: Channel multiplier in transitions.
        msa_dropout: Dropout rate for MSA activations.
        pair_dropout: Dropout rate for pair activations.
        inf: Safe infinity value.
        eps: Epsilon to prevent division by zero.
        eps_opm: Epsilon to prevent division by zero in outer product mean.
        chunk_size_msa_att: Optional chunk size for a batch-like dimension in MSA attention.
        chunk_size_opm: Optional chunk size for a batch-like dimension in outer product mean.
        chunk_size_tri_att: Optional chunk size for a batch-like dimension in triangular attention.

    """

    def __init__(
        self,
        c_e: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_tri_mul: int,
        c_hidden_tri_att: int,
        num_heads_msa: int,
        num_heads_tri: int,
        num_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        eps_opm: float,
        chunk_size_msa_att: Optional[int],
        chunk_size_opm: Optional[int],
        chunk_size_tri_att: Optional[int],
        block_size_tri_mul: Optional[int],
        outer_product_mean_first: bool = False,
    ) -> None:
        super().__init__()
        self.opm_first = outer_product_mean_first
        self.blocks = nn.ModuleList(
            [
                ExtraMSABlock(
                    c_e=c_e,
                    c_z=c_z,
                    c_hidden_msa_att=c_hidden_msa_att,
                    c_hidden_opm=c_hidden_opm,
                    c_hidden_tri_mul=c_hidden_tri_mul,
                    c_hidden_tri_att=c_hidden_tri_att,
                    num_heads_msa=num_heads_msa,
                    num_heads_tri=num_heads_tri,
                    transition_n=transition_n,
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                    inf=inf,
                    eps=eps,
                    eps_opm=eps_opm,
                    chunk_size_msa_att=chunk_size_msa_att,
                    chunk_size_opm=chunk_size_opm,
                    chunk_size_tri_att=chunk_size_tri_att,
                    block_size_tri_mul=block_size_tri_mul,
                    outer_product_mean_first=outer_product_mean_first,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        gradient_checkpointing: bool,
        inplace_safe: bool,
    ) -> torch.Tensor:
        """Extra MSA Stack forward pass.

        Args:
            m: [batch, N_extra_seq, N_res, c_e] extra MSA representation
            z: [batch, N_res, N_res, c_z] pair representation
            msa_mask: [batch, N_extra_seq, N_res] extra MSA mask
            pair_mask: [batch, N_res, N_res] pair mask
            gradient_checkpointing: whether to use gradient checkpointing

        Returns:
            z: [batch, N_res, N_res, c_z] updated pair representation

        """
        if gradient_checkpointing:
            assert torch.is_grad_enabled()
            z = self._forward_blocks_with_gradient_checkpointing(
                m=m,
                z=z,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
            )
        else:
            z = self._forward_blocks(
                m=m,
                z=z,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                inplace_safe=inplace_safe,
            )
        return z

    def _forward_blocks(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        inplace_safe: bool,
    ) -> torch.Tensor:
        if mp.is_enabled():
            msa_col_pad_size = get_pad_size(m, -2, mp.size())
            msa_row_pad_size = get_pad_size(m, -3, mp.size())
            m = pad_tensor(m, -2, msa_col_pad_size)
            m = pad_tensor(m, -3, msa_row_pad_size)
            if self.opm_first:
                m = mp.scatter(m, dim=-2)
            else:
                m = mp.scatter(m, dim=-3)
            msa_mask = pad_tensor(msa_mask, -1, msa_col_pad_size)
            msa_mask = pad_tensor(msa_mask, -2, msa_row_pad_size)
            pair_pad_size = get_pad_size(z, -3, mp.size())
            z = pad_tensor(z, -2, pair_pad_size)
            z = pad_tensor(z, -3, pair_pad_size)
            z = mp.scatter(z, dim=-3)
            pair_mask = pad_tensor(pair_mask, -1, pair_pad_size)
            pair_mask = pad_tensor(pair_mask, -2, pair_pad_size)

        for block in self.blocks:
            m, z = block(m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask, inplace_safe=inplace_safe)

        if mp.is_enabled():
            z = mp.gather(z, dim=-3)
            if pair_pad_size != 0:
                z = z[..., : z.size(-3) - pair_pad_size, : z.size(-2) - pair_pad_size, :]

        return z

    def _forward_blocks_with_gradient_checkpointing(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        blocks = [partial(block, msa_mask=msa_mask, pair_mask=pair_mask) for block in self.blocks]

        if mp.is_enabled():
            if self.opm_first:
                m = mp.scatter(m, dim=-2)
            else:
                m = mp.scatter(m, dim=-3)
            z = mp.scatter(z, dim=-3)

        for block in blocks:
            m, z = gradient_checkpointing_fn(block, m, z, use_reentrant=True)

        if mp.is_enabled():
            z = mp.gather(z, dim=-3)

        return z
