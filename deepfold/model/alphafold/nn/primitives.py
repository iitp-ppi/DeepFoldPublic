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
    trunc_normal_init_(weights, scale=1.0)


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
        Args:
            in_dim:
                size of each input sample
            out_dim:
                size of each output sample
            bias:
                If set to `False`, the layer will not learn an additive bias.
            init:
                The initializer to initialize the learnable weights and bias.
                Choose from:

                - `'default'`: LeCun fan-in truncated normal initialization
                - `'relu'`: He initialization with truncated normal distribution
                - `'glorot'`: fan-average Glorot uniform initialization
                - `'gating'`: weights=0, bias=1
                - `'normal'`: Normal initialization with `std=1/sqrt(fan_in)`
                - `'final'`: weights=0, bias=0

                Overridden by `init_fn` if not `None`.
            init_fn:
                A custom initializer taking weight and bias as inputs.

        Note:
            [From AF2 Suppl. Sec. 1.11.4]

            **`'Default'`** By default, the weights of the Linera layers are initialized using the LeCun (fan-in)
            truncated normal initialization strategy (LeCun, Y. et al., 1998) with a truncated normal distribution.
            By default, the biases of the Linear layers are initialized by zero.

            **`'relu'`** For the layers immediately followed by a ReLU activations, they are initialized with He
            initializer (He, K et al., 2015). It is a truncated normal distribution which has different scale.

            **`'glorot'`** The queries, keys and values projection layers in self-attention layerse are initialized
            using the 'fan-average' Glorot uniform scheme (Glorot X. et al., 2010) which also knwon as Xavier normal.

            **`'final'`** The weights of the *final* linear layers in every residual layer is initialized by zero.
            This is helpful for improving stability of training because this ensures that every residual layer acts as
            an identity operation at initialization.
            Furthermore, all the final projection weight layers of the network: masked MSA prediction logits, residue
            distance prediction logits, model confidence prediction logits is zero-initialized.

            **`'gating'`** Gating linear layers (`gating_w`), i.e., the Linear layers immediately followed by a sigmoid,
            are initialized with zero weights. The biases are initialized with a constant value of one, ensuring that
            at initialization, the gates are mostly-opened with a value of `sigmoid(1)`, approximately 0.73.
        """

        super().__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string")


class LayerNorm(nn.Module):
    def __init__(self, c_in: int, eps: float = 1e-05):
        super().__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype

        if dtype is torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(
                    x,
                    self.c_in,
                    self.weight.to(dtype=dtype),
                    self.bias.to(dtype=dtype),
                    self.eps,
                )
        else:
            out = nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )

        return out


@torch.jit.ignore
def softmax_no_cast(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax that do not automatically cast to float32 when the input is of type bfloat16.
    """

    dtype = x.dtype

    if dtype is torch.bfloat16:
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(x, dim=dim)
    else:
        s = torch.nn.functional.softmax(x, dim=dim)

    return s


class Attention(nn.Module):
    """
    Standard multi-head attention used in AlphaFold.
    Allows multiple bias vectors.
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        n_heads: int,
        gating: bool = True,
    ) -> None:
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per_head hidden dimension
            n_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super().__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.n_heads = n_heads
        self.gating = gating

        # c_hidden is not the per-head channel dimension.

        self.linear_q = Linear(self.c_q, self.c_hidden * self.n_heads, bias=False, init="glorot")
        self.linear_k = Linear(self.c_k, self.c_hidden * self.n_heads, bias=False, init="glorot")
        self.linear_v = Linear(self.c_v, self.c_hidden * self.n_heads, bias=False, init="glorot")
        self.linear_o = Linear(self.c_hidden * self.n_heads, self.c_q, init="final")
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(self.c_q * self.c_hidden * self.n_heads, init="gating")
        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self, q_x: torch.Tensor, kv_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, Q/K=V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/=V, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q /= math.sqrt(self.c_hidden)
