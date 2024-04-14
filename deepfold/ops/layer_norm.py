import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):

    def __init__(self, normalization_shape, eps=1e-5, elementwise_affine=True) -> None:
        super().__init__()
        if isinstance(normalization_shape, numbers.Integral):
            normalization_shape = (normalization_shape,)
        self.normalized_shape = torch.Size(normalization_shape)
        self.eps = eps
        assert elementwise_affine
        self.weight = nn.Parameter(torch.Tensor(*normalization_shape))
        self.bias = nn.Parameter(torch.Tensor(*normalization_shape))
        self.reset_parameters()

        def torch_layer_norm(input):
            return F.layer_norm(
                input,
                self.normalized_shape,
                self.weight.type(input.dtype),
                self.bias.type(input.dtype),
            )

        self.func = torch_layer_norm

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input)
