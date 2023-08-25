# DeepFold Team


import math
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm

from deepfold.utils.misc import prod
from deepfold.utils.precision import is_fp16_enabled
from deepfold.utils.tensor_utils import flatten_final_dims, permute_final_dims


def _calculate_fan(linear_weight_shape: torch.Size, fan: str = "fan_in") -> float:
    fan_out, fan_in = tuple(linear_weight_shape)

    if fan == "fan_in":
        f = float(fan_in)
    elif fan == "fan_out":
        f = float(fan_out)
    elif fan == "fan_avg":
        f = 0.5 * (fan_in + fan_out)
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights: torch.Tensor, scale: float = 1.0, fan: str = "fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale /= max(1, f)
    a = -2.0
    b = 2.0
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights: torch.Tensor):
    trunc_normal_init_(weights, scale=1)


def he_normal_init_(weights: torch.Tensor):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights: torch.Tensor):
    nn.init.xavier_normal_(weights, gain=1.0)


def final_init_(weights: torch.Tensor):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights: torch.Tensor):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights: torch.Tensor):
    nn.init.kaiming_normal_(weights, nonlinearity="linear")


# Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
def ipa_point_weights_init_(weights: torch.Tensor):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class Linear(nn.Linear):
    """A linear layer with non-standard initializations."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> None:
        """
        - "default": LeCun fan-in truncated normal initialization
        """
