import torch
import torch.nn as nn

from deepfold.model.v2.modules.linear import Linear


class BackboneUpdate(nn.Module):
    """Backbone Update module.

    Supplementary '1.8.3 Backbone update': Algorithm 23.

    Args:
        c_s: Single representation dimension (channels).

    Notes:
        [b, c, d, x, y, z]
    """

    def __init__(self, c_s: int) -> None:
        super(BackboneUpdate, self).__init__()
        self.linear = Linear(c_s, 6, bias=True, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(s)
