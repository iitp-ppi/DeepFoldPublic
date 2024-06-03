from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpointing_fn

import deepfold.distributed.model_parallel as mp
from deepfold.modules.layer_norm import LayerNorm
from deepfold.modules.template_pair_block import TemplatePairBlock
from deepfold.utils.dist_utils import get_pad_size, pad_tensor


class TemplatePairStack(nn.Module):
    """Template Pair Stack module.

    Supplementary '1.7.1 Template stack': Algorithm 16.

    Args:
        c_t: Template representation dimension (channels).
        c_hidden_tri_att: Hidden dimension in triangular attention.
        c_hidden_tri_mul: Hidden dimension in multiplicative updates.
        num_blocks: Number of blocks in the stack.
        num_heads_tri: Number of heads used in triangular attention.
        pair_transition_n: Channel multiplier in pair transition.
        dropout_rate: Dropout rate for pair activations.
        inf: Safe infinity value.
        chunk_size_tri_att: Optional chunk size for a batch-like dimension
            in triangular attention.

    """

    def __init__(
        self,
        c_t: int,
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        num_blocks: int,
        num_heads_tri: int,
        pair_transition_n: int,
        dropout_rate: float,
        inf: float,
        chunk_size_tri_att: Optional[int],
        block_size_tri_mul: Optional[int],
        tri_att_first: bool = True,
    ) -> None:
        super().__init__()
        self.tri_att_first = tri_att_first
        self.blocks = nn.ModuleList(
            [
                TemplatePairBlock(
                    c_t=c_t,
                    c_hidden_tri_att=c_hidden_tri_att,
                    c_hidden_tri_mul=c_hidden_tri_mul,
                    num_heads_tri=num_heads_tri,
                    pair_transition_n=pair_transition_n,
                    dropout_rate=dropout_rate,
                    inf=inf,
                    chunk_size_tri_att=chunk_size_tri_att,
                    block_size_tri_mul=block_size_tri_mul,
                    tri_att_first=tri_att_first,
                )
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm = LayerNorm(c_t)

    def forward(
        self,
        t: torch.Tensor,
        mask: torch.Tensor,
        gradient_checkpointing: bool,
        inplace_safe: bool,
    ) -> torch.Tensor:
        """Template Pair Stack forward pass.

        Args:
            t: [batch, N_templ, N_res, N_res, c_t] template representation
            mask: [batch, N_res, N_res] pair mask
            gradient_checkpointing: whether to use gradient checkpointing

        Returns:
            t: [batch, N_templ, N_res, N_res, c_t] updated template representation

        """
        if gradient_checkpointing:
            assert torch.is_grad_enabled()
            t = self._forward_blocks_with_gradient_checkpointing(t=t, mask=mask)
        else:
            t = self._forward_blocks(t=t, mask=mask, inplace_safe=inplace_safe)
        t = self.layer_norm(t)
        return t

    def _forward_blocks(
        self,
        t: torch.Tensor,
        mask: torch.Tensor,
        inplace_safe: bool,
    ) -> torch.Tensor:
        if mp.is_enabled():
            pad_size = get_pad_size(t, -2, mp.size())
            t = pad_tensor(t, -2, pad_size)
            t = pad_tensor(t, -3, pad_size)
            t = mp.scatter(t, dim=-3)
            mask = pad_tensor(mask, -1, pad_size)
            mask = pad_tensor(mask, -2, pad_size)

        for block in self.blocks:
            t = block(t=t, mask=mask, inplace_safe=inplace_safe)

        if mp.is_enabled():
            t = mp.gather(t, dim=-3)
            if pad_size != 0:
                t = t[..., :, : t.size(-3) - pad_size, : t.size(-2) - pad_size, :]

        return t

    def _forward_blocks_with_gradient_checkpointing(
        self,
        t: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        blocks = [partial(block, mask=mask) for block in self.blocks]

        if mp.is_enabled():
            t = mp.scatter(t, dim=-3)

        for block in blocks:
            t = gradient_checkpointing_fn(block, t, use_reentrant=True)

        if mp.is_enabled():
            t = mp.gather(t, dim=-3)

        return t
