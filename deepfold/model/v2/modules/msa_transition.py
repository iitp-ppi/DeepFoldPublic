from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.model.v2.modules.inductor as inductor
from deepfold.model.v2.modules.layer_norm import LayerNorm
from deepfold.model.v2.modules.linear import Linear
from deepfold.utils.iter_utils import slice_generator


class MSATransition(nn.Module):
    """MSA Transition module.

    Supplementary '1.6.3 MSA transition': Algorithm 9.

    Args:
        c_m: MSA (or Extra MSA) representation dimension (channels).
        n: `c_m` multiplier to obtain hidden dimension (channels).

    """

    def __init__(
        self,
        c_m: int,
        n: int,
    ) -> None:
        super().__init__()
        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, n * c_m, bias=True, init="relu")
        self.linear_2 = Linear(n * c_m, c_m, bias=True, init="final")

    # TODO: Chunk
    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """MSA Transition forward pass.

        Args:
            m: [batch, N_seq, N_res, c_m] MSA representation
            mask: [batch, N_seq, N_res] MSA mask

        Returns:
            m: [batch, N_seq, N_res, c_m] updated MSA representation

        """
        # NOTE: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        if inductor.is_enabled():
            forward_fn = _forward_jit
        elif inductor.is_enabled_and_autograd_off():
            forward_fn = _forward_jit
        else:
            forward_fn = _forward_eager
        return forward_fn(
            self.layer_norm(m),
            mask,
            self.linear_1.weight,
            self.linear_1.bias,
            self.linear_2.weight,
            self.linear_2.bias,
            m,
        )


def _forward_eager(
    m: torch.Tensor,
    mask: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m = F.linear(m, w1, b1)
    m = torch.relu(m)
    m = F.linear(m, w2, b2)
    m = m * mask
    m = out + m
    return m


_forward_jit = torch.compile(_forward_eager)
