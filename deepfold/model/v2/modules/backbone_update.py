import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.model.v2.modules.inductor as inductor
from deepfold.model.v2.modules.linear import Linear


class BackboneUpdate(nn.Module):
    """Backbone Update module.

    Supplementary '1.8.3 Backbone update': Algorithm 23.

    Args:
        c_s: Single representation dimension (channels).

    """

    def __init__(self, c_s: int) -> None:
        super().__init__()
        self.linear = Linear(c_s, 6, bias=True, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if inductor.is_enabled_on_hopper() and dap.size() in {2, 8}:
            forward_fn = _forward_jit
        else:
            forward_fn = _forward_eager
        return forward_fn(s, self.linear.weight, self.linear.bias)


def _forward_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, w, b)


_forward_jit = torch.compile(_forward_eager)
