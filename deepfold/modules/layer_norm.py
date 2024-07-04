import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.modules.inductor as inductor


class LayerNorm(nn.Module):
    """Layer Normalization module.

    Supplementary '1.11.4 Parameters initialization': Layer normalization.

    Args:
        in_channels: Last dimension of the input tensor.
        eps: A value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.normalized_shape = (in_channels,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))

        self._ln_eager_func = F.layer_norm
        # self._ln_inductor_func = torch.compile(F.layer_norm)
        self._ln_inductor_func = self._ln_eager_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or inductor.is_enabled():
            return self._ln_inductor_func(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            return self._ln_eager_func(x, self.normalized_shape, self.weight, self.bias, self.eps)
