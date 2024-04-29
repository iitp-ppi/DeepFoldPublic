from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.distributed.model_parallel as mp
import deepfold.modules.inductor as inductor


class Dropout(nn.Module):
    """Dropout module.

    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.

    Supplementary '1.11.6 Dropout details'.

    Args:
        p: Dropout rate (probability of an element to be zeroed).
        share_dim: Dimension(s) along which the dropout mask is shared.
        inplace: If set to `True`, will do this operation in-place.

    """

    def __init__(
        self,
        p: float,
        share_dim: Union[int, Tuple[int, ...]] = (),
        inplace: bool = False,
    ) -> None:
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.p = p
        if type(share_dim) == int:
            share_dim = (share_dim,)
        else:
            assert isinstance(share_dim, tuple)
        self.share_dim = share_dim
        self.inplace = inplace

    def forward(
        self,
        x: torch.Tensor,
        add_output_to: torch.Tensor,
        scattered_dim: Optional[int] = None,
    ) -> torch.Tensor:
        shape = list(x.shape)
        for d in self.share_dim:
            shape[d] = 1
        if scattered_dim is not None:
            shape[scattered_dim] *= mp.rank()
        mask = x.new_ones(shape)
        mask = F.dropout(
            input=mask,
            p=self.p,
            training=self.training,
            inplace=self.inplace,
        )
        if scattered_dim is not None:
            mask = mp.scatter(mask, dim=scattered_dim)
        x = _mul_add(x, mask, add_output_to)
        return x


class DropoutRowwise(Dropout):
    """Dropout Rowwise module."""

    def __init__(self, p: float) -> None:
        super().__init__(p=p, share_dim=-3)


class DropoutColumnwise(Dropout):
    """Dropout Columnwise module."""

    def __init__(self, p: float) -> None:
        super().__init__(p=p, share_dim=-2)


def _mul_add_eager(
    x: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return y + x * mask


_mul_add_jit = torch.compile(_mul_add_eager)


def _mul_add(
    x: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    if inductor.is_enabled():
        mul_add_fn = _mul_add_jit
    else:
        mul_add_fn = _mul_add_eager
    return mul_add_fn(x, mask, y)
