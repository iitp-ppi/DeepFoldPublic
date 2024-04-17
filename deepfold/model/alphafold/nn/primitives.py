# DeepFold Team


import math
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm

from deepfold.utils.tensor_utils import flatten_final_dims, permute_final_dims


def prod(x: Sequence[int]) -> int:
    return math.prod(x)


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
    """
    A layer normalization layer.
    """

    def __init__(self, c_in: int, eps: float = 1e-05):
        """
        Args:
            c_in:
                The number of channels
            eps:
                Small epsilon to avoid division by zero variance.

        Note:
            The scales are initialized with a constant value of one.
            The offsets are zero-initialized.
        """

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


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: List[torch.Tensor],
) -> torch.Tensor:
    # [*, H, C, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax_no_cast(a, -1)

    # [*, H, Q, C]
    a = torch.matmul(a, value)

    return a


class Attention(nn.Module):
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        c_output: int,
        num_heads: int,
        gating: bool = True,
    ) -> None:
        """
        Args:
            c_q:
                Input dimension of query data ($C_Q$)
            c_k:
                Input dimension of key data ($C_K$)
            c_v:
                Input dimension of value data ($C_V$)
            c_hidden:
                Per head hidden dimension ($C$)
            c_output:
                Output dimension ($C_O$)
            num_heads:
                Number of attention heads ($H$)
            gating:
                Whether the output should be gated using query data
        """
        super().__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.c_output = c_output
        self.num_heads = num_heads
        self.gating = gating

        # c_hidden is not the per-head channel dimension.

        self.linear_q = Linear(self.c_q, self.c_hidden * self.num_heads, bias=False, init="glorot")
        self.linear_k = Linear(self.c_k, self.c_hidden * self.num_heads, bias=False, init="glorot")
        self.linear_v = Linear(self.c_v, self.c_hidden * self.num_heads, bias=False, init="glorot")

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(self.c_q, self.c_hidden * self.num_heads, init="gating")

        self.linear_o = Linear(self.c_hidden * self.num_heads, self.c_output, init="final")
        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self, q_x: torch.Tensor, kv_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, Q/K=V, H * C]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K=V, H, C]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))
        k = k.view(k.shape[:-1] + (self.num_heads, -1))
        v = v.view(v.shape[:-1] + (self.num_heads, -1))

        # [*, H, Q/K=V, C]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        # Normalize
        q *= self.c_hidden ** (-0.5)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.linear_g(q_x)
            g = self.sigmoid(g)

            # [*, Q, H, C]
            g = g.view(g.shape[:-1] + (self.num_heads, -1))
            o = o * g

        # [*, Q, H * C]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                $(*, Q, C_Q)$ A tensor of queries
            kv_x:
                $(*, K, C_K)$ A tensor of memories from which the keys and values are projected
            biases:
                List of biases whose shape is $(*, H, Q, K)$

        Returns:
            $(*, Q, C_O)$ A tensor of attention update

        Note:
            Biases valued with `-inf` prevents attention to corresponding positions.
            Therefore the layer doesn't have an attention mask argument.
        """
        q, k, v = self._prep_qkv(q_x, kv_x)

        o = _attention(q, k, v, biases)  # [*, H, Q, C]
        o = o.transpose(-2, -3)  # [*, Q, H, C]
        o = self._wrap_up(o, q_x)  # [*, Q, C_output]

        return o


class GlobalAttention(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        c_output: int,
        num_heads: int,
        inf: float,
        eps: float,
    ) -> None:
        """
        Args:
            c_in:
                Dimension of input channel
            c_hidden:
                Dimension of hidden channel
            c_output:
                Dimension of output channel
            num_heads:
                Number of heads
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.c_output = c_output
        self.num_heads = num_heads
        self.inf = inf
        self.eps = eps

        self.linear_q = Linear(self.c_in, self.c_hidden * self.num_heads, bias=False, init="glorot")
        self.linear_k = Linear(self.c_in, self.c_hidden, bias=False, init="glorot")
        self.linear_v = Linear(self.c_in, self.c_hidden, bias=False, init="glorot")
        self.linear_g = Linear(self.c_in, self.c_hidden * self.num_heads, init="gating")
        self.linear_o = Linear(self.c_hidden * self.num_heads, self.c_in, init="final")
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            m: [*, N, S, C_m]
            mask: [*, N, S]
        Return: [*, N, S, C_m]
        """
        # [*, N, C_m]
        q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (torch.sum(mask, dim=-1)[..., None] + self.eps)

        # [*, N, H * C]
        q = self.linear_q(q)
        q *= self.c_hidden ** (-0.5)

        # [*, N, H, C]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))

        # [*, N, S, C]
        k = self.linear_k(m)
        v = self.linear_v(m)

        # [*, N, 1, S]
        bias = (self.inf * (mask - 1))[..., :, None, :]

        # # [*, N, H, S]
        # a = torch.matmul(q, k.transpose(-1, -2))  # [*, N, H, C] @ [*, N, C, S]
        # a += bias
        # a = softmax_no_cast(a)

        # #  [*, N, H, C]
        # o = torch.matmul(a, v)  # [*, N, H, S] @ [*, N, S, C]

        o = _attention(q, k, v, [bias])

        # [*, N, S, H * C]
        g = self.sigmoid(self.linear_g(m))
        # [*, N, S, H, C]
        g = g.view(g.shape[:-1] + (self.num_heads, -1))

        # [*, N, S, H, C]
        o = o.unsqueeze(-3) * g  # [*, N, 1, H, C] * [*, N, S, H, C]

        # [*, N, S, H * C]
        o = flatten_final_dims(o, 2)

        # [*, N, S, C_m]
        m = self.linear_o(o)

        return m
