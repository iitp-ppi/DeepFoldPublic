from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold.core.ops import LayerNorm, evoformer_attention
from deepfold.utils.tensor_utils import flatten_final_dims, permute_final_dims

__all__ = [
    "LayerNorm",
    "Linear",
    "residual",
    "fused_bias_dropout_add_training",
    "fused_bias_dropout_add_inference",
]


class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
        prec: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError(f"Invalid init method: '{init}'")

        self.prec = prec

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        in_dtype = input.dtype

        if self.prec is not None:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=self.prec) if self.bias is not None else None
                return nn.functional.linear(
                    input.to(dtype=self.prec),
                    self.weight.to(dtype=self.prec),
                    bias,
                ).to(dtype=in_dtype)

        return nn.functional.linear(input, self.weight, self.bias)

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0.0, scale=1.0)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")


def residual(residual: torch.Tensor, x: torch.Tensor, is_training: bool):
    if is_training:
        return x + residual
    else:
        residual += x
        return residual


@torch.jit.script
def fused_bias_dropout_add_training(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    dropmask: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    return (x + bias) * F.dropout(dropmask, p=prob, training=True) + residual


@torch.jit.script
def fused_bias_dropout_add_inference(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    residual += bias + x
    return residual


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of type bfloat16.
    """

    if t.dtype is torch.bfloat16:
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: List[torch.Tensor],
) -> torch.Tensor:

    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


class Attention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = True,
        mode: str = "naive",
    ) -> None:

        super().__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating

        self.linear_q = Linear(q_dim, total_dim, bias=False, init="glorot")
        self.linear_k = Linear(k_dim, total_dim, bias=False, init="glorot")
        self.linear_v = Linear(v_dim, total_dim, bias=False, init="glorot")
        self.linear_o = Linear(total_dim, q_dim, init="final")

        # Gating
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(q_dim, total_dim, init="gating")

        # Compute the normalization factor
        self.norm = head_dim**-0.5

        def naive_attn(q, k, v, b):
            o = _attention(q, k, v, b)
            return o.transpose(-2, -3)

        def evo_attn(q, k, v, b):
            assert len(b) <= 2
            return evoformer_attention(q, k, v, b)

        if self.mode == "naive":
            self.kernel = naive_attn
        elif self.mode == "evo":
            self.kernel = evo_attn
            self.norm = None
        else:
            raise ValueError(f"Not supported attention mode: {mode}")

    def _prep_qkv(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        apply_scale: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q *= self.norm

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:

        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
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

        if biases is None:
            biases = []

        q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=True if self.norm is None else False)
        o = self.kernel(q, k, v, biases)
        o = self._wrap_up(o, q_x)

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
