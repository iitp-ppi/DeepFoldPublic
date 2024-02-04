from typing import Optional

import torch
import torch.nn.functional as F


def softmax_dropout(
    input: torch.Tensor,
    dropout_prob: float,
    is_training: bool = True,
    mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    inplace: bool = True,
) -> torch.Tensor:

    input = input.contiguous()

    if not inplace:
        input = input.clone()

    if mask is not None:
        input += mask

    if bias is not None:
        input += bias

    return F.dropout(F.softmax(input, dim=-1), p=dropout_prob, training=is_training)
