from typing import Optional

import torch
import torch.nn as nn

from deepfold.model.v2.modules.layer_norm import LayerNorm
from deepfold.model.v2.modules.linear import Linear
from deepfold.utils.chunk_utils import chunk_layer


class SingleTransition(nn.Module):
    """Single Transition module.

    Supplementary '1.8 Structure module': Algorithm 20, lines 8-9.

    Args:
        c_s: Single representation dimension (channels).
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        c_s: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.linear_1 = Linear(c_s, c_s, bias=True, init="relu")
        self.linear_2 = Linear(c_s, c_s, bias=True, init="relu")
        self.linear_3 = Linear(c_s, c_s, bias=True, init="final")
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm(c_s)

    def forward(
        self,
        s: torch.Tensor,
    ) -> torch.Tensor:
        s = s + self.linear3(torch.relu(self.linear2(torch.relu(self.linear1(s)))))
        s = self.layer_norm(self.dropout(s))
        return s
