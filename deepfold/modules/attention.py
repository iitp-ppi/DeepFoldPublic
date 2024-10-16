import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.modules.inductor as inductor
from deepfold.modules.linear import Linear
from deepfold.modules.tweaks import evo_attn
from deepfold.utils.iter_utils import slice_generator


def _attention_gate_eager(
    output: torch.Tensor,
    gate: torch.Tensor,
    linear_g_bias: torch.Tensor,
) -> torch.Tensor:
    gate = torch.sigmoid(gate + linear_g_bias)
    output = output * gate
    return output


_attention_gate_jit = torch.compile(_attention_gate_eager)


class SelfAttentionWithGate(nn.Module):
    """Self Multi-Head Attention module with gating.

    Args:
        c_qkv: Input dimension of query|key|value data tensor (channels).
        c_hidden: Hidden dimension (per-head).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.
            Supplementary '1.11.8 Reducing the memory consumption': Inference.

    """

    def __init__(
        self,
        c_qkv: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        chunk_size: Optional[int],
        impl: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.c_qkv = c_qkv
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.inf = inf
        self.chunk_size = chunk_size
        self.impl = impl

        total_dim = c_hidden * num_heads
        self.linear_q = Linear(c_qkv, total_dim, bias=False, init="glorot")
        self.linear_k = Linear(c_qkv, total_dim, bias=False, init="glorot")
        self.linear_v = Linear(c_qkv, total_dim, bias=False, init="glorot")
        self.linear_g = Linear(c_qkv, total_dim, init="gating")
        self.linear_o = Linear(c_hidden * num_heads, c_qkv, bias=True, init="final")

        try:
            from deepfold_kernels.evoformer_attn import DS4Sci_EvoformerAttention
        except ModuleNotFoundError:
            from deepfold.modules.tweaks import evo_attn

            # Disable evoformer attention
            evo_attn.disable()

    def forward(
        self,
        input_qkv: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor],
        add_transposed_output_to: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention forward pass.

        Args:
            input_qkv: [*, QKV, c_qkv] query data (QKV == Q == K == V)
            mask: Logit mask tensor broadcastable to [*, num_heads, Q, K]
            bias: Optional logit bias tensor broadcastable to [*, num_heads, Q, K]
            add_transposed_output_to:
                Optional tensor to which transposed output will be added elementwisely.

        Returns:
            output: [*, Q, c_qkv] tensor

        """
        query, key, value, gate = self._prep_qkvg(input_qkv)
        # query: [*, num_heads, Q, c_hidden]
        # key:   [*, num_heads, K, c_hidden]
        # value: [*, num_heads, V, c_hidden]
        # gate:  [*, Q, num_heads * c_hidden]

        if evo_attn.is_enabled():
            impl = "evo"
        else:
            impl = "torch"

        if self.impl is not None:
            impl = self.impl

        if impl == "torch":
            output = self._attention_forward(query, key, value, mask, bias)
            # output: [*, num_heads, Q, c_hidden] for torch implementation.
            output = output.transpose(-2, -3)
            # output: [*, Q, num_heads, c_hidden]
        elif impl == "evo":
            from deepfold.ops.evoformer_attention import deepspeed_evo_attn

            mask_bias = (mask - 1.0) * self.inf
            biases = [mask_bias]
            if bias is not None:
                biases.append(bias)
            # output = deepspeed_evo_attn(query / math.sqrt(self.c_hidden), key, value, biases)
            output = deepspeed_evo_attn(query, key, value, biases)
        else:
            raise ValueError(f"Unsupported implementation '{impl}'")

        del query, key, value

        output = output.reshape(*output.shape[:-2], self.num_heads * self.c_hidden)
        # output = output.reshape(output.shape[:-2] + (self.num_heads * self.c_hidden,))
        # output: [*, Q, num_heads * c_hidden]

        if self.training:
            output = _attention_gate_jit(output, gate, self.linear_g.bias)
        else:
            output = _attention_gate_eager(output, gate, self.linear_g.bias)
        # output: [*, Q, num_heads * c_hidden]

        if add_transposed_output_to is None:
            output = self.linear_o(output)
        else:
            output = _linear_transpose_add(
                output,
                self.linear_o.weight,
                self.linear_o.bias,
                add_transposed_output_to,
            )
        # output: [*, Q, c_qkv]

        return output

    def _prep_qkvg(
        self,
        input_qkv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_qkv: [*, QKV, c_qkv]

        q = self.linear_q(input_qkv)
        k = self.linear_k(input_qkv)
        v = self.linear_v(input_qkv)
        g = self.linear_g(input_qkv)
        # q: [*, Q, num_heads * c_hidden]
        # k: [*, K, num_heads * c_hidden]
        # v: [*, V, num_heads * c_hidden]
        # g: [*, Q, num_heads * c_hidden]

        q = q.view(q.shape[:-1] + (self.num_heads, self.c_hidden))
        k = k.view(k.shape[:-1] + (self.num_heads, self.c_hidden))
        v = v.view(v.shape[:-1] + (self.num_heads, self.c_hidden))
        # q: [*, Q, num_heads, c_hidden]
        # k: [*, K, num_heads, c_hidden]
        # v: [*, V, num_heads, c_hidden]

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        # q: [*, num_heads, Q, c_hidden]
        # k: [*, num_heads, K, c_hidden]
        # v: [*, num_heads, V, c_hidden]

        # q = q / math.sqrt(self.c_hidden) scaling moved to _attention function

        return q, k, v, g

    def _attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.chunk_size is None:
            if inductor.is_enabled():
                return _attention_eager(query, key, value, mask, bias, self.inf)
            else:
                return _attention_jit(query, key, value, mask, bias, self.inf)
        else:
            return _attention_chunked(query, key, value, mask, bias, self.inf, self.chunk_size)


class CrossAttentionNoGate(nn.Module):
    """Cross Multi-Head Attention module without gating.

    Args:
        c_q: Input dimension of query data tensor (channels).
        c_kv: Input dimension of key|value data tensor (channels).
        c_hidden: Hidden dimension (per-head).
        num_heads: Number of attention heads.
        inf: Safe infinity value.
        chunk_size: Optional chunk size for a batch-like dimension.
            Supplementary '1.11.8 Reducing the memory consumption': Inference.

    """

    def __init__(
        self,
        c_q: int,
        c_kv: int,
        c_hidden: int,
        num_heads: int,
        inf: float,
        chunk_size: Optional[int],
        impl: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.c_q = c_q
        self.c_kv = c_kv
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.inf = inf
        self.chunk_size = chunk_size
        self.impl = impl

        self.linear_q = Linear(c_q, c_hidden * num_heads, bias=False, init="glorot")
        self.linear_k = Linear(c_kv, c_hidden * num_heads, bias=False, init="glorot")
        self.linear_v = Linear(c_kv, c_hidden * num_heads, bias=False, init="glorot")
        self.linear_o = Linear(c_hidden * num_heads, c_q, bias=True, init="final")

    def forward(
        self,
        input_q: torch.Tensor,
        input_kv: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Attention forward pass.

        Args:
            input_q: [*, Q, c_q] query data
            input_kv: [*, KV, c_kv] key|value data (KV == K == V)
            mask: Logit mask tensor broadcastable to [*, num_heads, Q, KV]
            bias: Optional logit bias tensor broadcastable to [*, num_heads, Q, KV]

        Returns:
            output: [*, Q, c_q] tensor

        """
        query, key, value = self._prep_qkv(input_q, input_kv)
        # query: [*, num_heads, Q, c_hidden]
        # key:   [*, num_heads, K, c_hidden]
        # value: [*, num_heads, V, c_hidden]
        if evo_attn.is_enabled():
            impl = "evo"
        else:
            impl = "torch"

        if self.impl is not None:
            impl = self.impl

        if impl == "torch":
            output = self._attention_forward(query, key, value, mask, bias)
            # output: [*, num_heads, Q, c_hidden] for torch implementation.
            output = output.transpose(-2, -3)
            # output: [*, Q, num_heads, c_hidden]
        elif impl == "evo":
            from deepfold.ops.evoformer_attention import deepspeed_evo_attn

            mask_bias = (mask - 1.0) * self.inf
            biases = [mask_bias]
            if bias is not None:
                biases.append(bias)
            # output = deepspeed_evo_attn(query / math.sqrt(self.c_hidden), key, value, biases)
            output = deepspeed_evo_attn(query, key, value, biases)
        else:
            raise ValueError(f"Unsupported implementation '{impl}'")

        del query, key, value

        output = output.reshape(output.shape[:-2] + (self.num_heads * self.c_hidden,))
        # output: [*, Q, num_heads * c_hidden]

        output = self.linear_o(output)
        # output: [*, Q, c_q]

        return output

    def _prep_qkv(
        self,
        input_q: torch.Tensor,
        input_kv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_q: [*, Q, c_q]
        # input_kv: [*, KV, c_kv]

        q = self.linear_q(input_q)
        k = self.linear_k(input_kv)
        v = self.linear_v(input_kv)
        # q: [*, Q, num_heads * c_hidden]
        # k: [*, K, num_heads * c_hidden]
        # v: [*, V, num_heads * c_hidden]

        q = q.view(q.shape[:-1] + (self.num_heads, self.c_hidden))
        k = k.view(k.shape[:-1] + (self.num_heads, self.c_hidden))
        v = v.view(v.shape[:-1] + (self.num_heads, self.c_hidden))
        # q: [*, Q, num_heads, c_hidden]
        # k: [*, K, num_heads, c_hidden]
        # v: [*, V, num_heads, c_hidden]

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        # q: [*, num_heads, Q, c_hidden]
        # k: [*, num_heads, K, c_hidden]
        # v: [*, num_heads, V, c_hidden]

        # q = q / math.sqrt(self.c_hidden) scaling moved to _attention function

        return q, k, v

    def _attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.chunk_size is None:
            if inductor.is_enabled():
                return _attention_eager(query, key, value, mask, bias, self.inf)
            else:
                return _attention_jit(query, key, value, mask, bias, self.inf)
        else:
            return _attention_chunked(query, key, value, mask, bias, self.inf, self.chunk_size)


def _attention_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    bias: Optional[torch.Tensor],
    inf: float,
) -> torch.Tensor:
    # query:  [*, num_heads, Q, c_hidden]
    # key:    [*, num_heads, K, c_hidden]
    # value:  [*, num_heads, V, c_hidden]
    # mask:   Logit mask tensor broadcastable to [*, num_heads, Q, K]
    # bias:   Optional logit bias tensor broadcastable to [*, num_heads, Q, K]
    # inf:    Safe infinity value.
    # assuming K == V

    key = torch.swapdims(key, -2, -1)
    # key: [*, num_heads, c_hidden, K]

    scaling = 1.0 / math.sqrt(query.size(-1))
    a = torch.matmul(query * scaling, key)
    # a: [*, num_heads, Q, K]

    a += (mask - 1.0) * inf
    # a: [*, num_heads, Q, K]

    if bias is not None:
        a += bias
    # a: [*, num_heads, Q, K]

    a = torch.softmax(a, dim=-1)
    # a: [*, num_heads, Q, K]

    a = torch.matmul(a, value)
    # a: [*, num_heads, Q, c_hidden]

    return a


_attention_jit = torch.compile(_attention_eager)


def _attention_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    bias: Optional[torch.Tensor],
    inf: float,
    chunk_size: int,
) -> torch.Tensor:
    output_chunks = []
    subbatch_size = query.size(1)
    if inductor.is_enabled():
        attention_fn = _attention_jit
    else:
        attention_fn = _attention_eager
    for left, right in slice_generator(0, subbatch_size, chunk_size):
        query_chunk = query[:, left:right]
        key_chunk = key[:, left:right]
        value_chunk = value[:, left:right]
        mask_chunk = mask[:, left:right] if mask.size(1) == subbatch_size else mask
        bias_chunk = bias[:, left:right] if bias is not None and bias.size(1) == subbatch_size else bias
        output_chunk = attention_fn(
            query=query_chunk,
            key=key_chunk,
            value=value_chunk,
            mask=mask_chunk,
            bias=bias_chunk,
            inf=inf,
        )
        output_chunks.append(output_chunk)
    return torch.cat(output_chunks, dim=1)


def _linear_transpose_add_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return y + F.linear(x, w, b).transpose(-2, -3)


_linear_transpose_add_jit = torch.compile(_linear_transpose_add_eager)


def _linear_transpose_add(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    if inductor.is_enabled():
        linear_transpose_add_fn = _linear_transpose_add_jit
    else:
        linear_transpose_add_fn = _linear_transpose_add_eager
    return linear_transpose_add_fn(x, w, b, y)
