from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.modules.inductor as inductor
from deepfold.modules.layer_norm import LayerNorm
from deepfold.modules.linear import Linear


class PairTransition(nn.Module):
    """Pair Transition module.

    Supplementary '1.6.7 Transition in the pair stack': Algorithm 15.

    Args:
        c_z: Pair or template representation dimension (channels).
        n: `c_z` multiplier to obtain hidden dimension (channels).

    """

    def __init__(
        self,
        c_z: int,
        n: int,
    ) -> None:
        super().__init__()
        self.layer_norm = LayerNorm(c_z)
        self.linear_1 = Linear(c_z, n * c_z, bias=True, init="relu")
        self.linear_2 = Linear(n * c_z, c_z, bias=True, init="final")

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Pair Transition forward pass.

        Args:
            z: [batch, N_res, N_res, c_z] pair representation
            mask: [batch, N_res, N_res] pair mask

        Returns:
            z: [batch, N_res, N_res, c_z] updated pair representation

        """
        # NOTE: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)

        input_z = z
        z = self.layer_norm(z)

        # make inductor happy - but why? what is the problem with original shape?
        original_shape = z.shape
        z = z.view(-1, z.shape[-1])

        if inductor.is_enabled():
            linear_relu_fn = _linear_relu_jit
        else:
            linear_relu_fn = _linear_relu_eager
        z = linear_relu_fn(z, self.linear_1.weight, self.linear_1.bias)

        if inductor.is_enabled():
            linear_view_add_fn = _linear_view_add_jit
        else:
            linear_view_add_fn = _linear_view_add_eager
        z = linear_view_add_fn(
            z,
            mask,
            self.linear_2.weight,
            self.linear_2.bias,
            input_z,
            inplace_safe,
        )

        z = z.view(original_shape)
        return z


def _linear_relu_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.relu(F.linear(x, w, b))


_linear_relu_jit = torch.compile(_linear_relu_eager)


def _linear_view_add_eager(
    z: torch.Tensor,
    mask: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    inplace: bool,
) -> torch.Tensor:
    z = F.linear(z, w, b)
    z = z.view(out.shape)
    if inplace:
        z *= mask
        z += out
    else:
        z = z * mask
        z = out + z
    return z


_linear_view_add_jit = torch.compile(_linear_view_add_eager)


# TODO: switch to this if possible:
# def _forward_eager(
#     z: torch.Tensor,
#     w1: torch.Tensor,
#     b1: torch.Tensor,
#     w2: torch.Tensor,
#     b2: torch.Tensor,
#     out: torch.Tensor,
# ) -> torch.Tensor:
#     z = F.linear(z, w1, b1)
#     z = torch.relu(z)
#     z = F.linear(z, w2, b2)
#     z = out + z
#     return z
# _forward_jit = torch.compile(_forward_eager)


# TODO: Chunk
