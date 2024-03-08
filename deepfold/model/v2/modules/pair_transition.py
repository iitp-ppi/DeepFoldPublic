from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.model.v2.modules.inductor as inductor
from deepfold.model.v2.modules.layer_norm import LayerNorm
from deepfold.model.v2.modules.linear import Linear
from deepfold.utils.iter_utils import slice_generator


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
        super(PairTransition, self).__init__()
        self.layer_norm = LayerNorm(c_z)
        self.linear_1 = Linear(c_z, n * c_z, bias=True, init="relu")
        self.linear_2 = Linear(n * c_z, c_z, bias=True, init="final")

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pair Transition forward pass.

        Args:
            z: [batch, N_res, N_res, c_z] pair representation
            mask: [batch, N_res, N_res] pair mask

        Returns:
            z: [batch, N_res, N_res, c_z] updated pair representation

        """
        # DeepMind forgets to apply the pair mask here.
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
        z = linear_view_add_fn(z, self.linear_2.weight, self.linear_2.bias, input_z)

        z = z.view(original_shape)
        return z

    # def forward(
    #     self,
    #     z: torch.Tensor,
    #     mask: torch.Tensor,
    # ) -> torch.Tensor:
    #     """Pair Transition forward pass.

    #     Args:
    #         z: [batch, N_res, N_res, c_z] pair representation
    #         mask: [batch, N_res, N_res] pair mask

    #     Returns:
    #         z: [batch, N_res, N_res, c_z] updated pair representation

    #     """
    #     # DeepMind forgets to apply the pair mask here.
    #     # TODO: why can't we just use this code which is similar to MSA transition?
    #     if inductor.is_enabled_on_ampere():
    #         forward_fn = _forward_jit
    #     elif inductor.is_enabled_on_hopper() and dap.size() in {2, 8}:
    #         forward_fn = _forward_jit
    #     elif inductor.is_enabled_on_hopper_and_autograd_off():
    #         forward_fn = _forward_jit
    #     else:
    #         forward_fn = _forward_eager
    #     return forward_fn(
    #         self.layer_norm(z),
    #         self.linear_1.weight,
    #         self.linear_1.bias,
    #         self.linear_2.weight,
    #         self.linear_2.bias,
    #         z,
    #     )


def _linear_relu_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.relu(F.linear(x, w, b))


_linear_relu_jit = torch.compile(_linear_relu_eager)


def _linear_view_add_eager(
    z: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    z = F.linear(z, w, b)
    z = z.view(out.shape)
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
