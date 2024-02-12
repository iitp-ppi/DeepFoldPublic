from math import prod
from typing import List, Optional, Tuple

import torch
from deepfold_kernel import evoformer_attention_bwd, evoformer_attention_fwd


def _attention_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert query.shape[-3] > 16, "seq_len must be greater than 16"
    out = torch.empty_like(query, dtype=query.dtype)
    nh = query.shape[-2]
    nq = (query.shape[-3] + 31) // 32 * 32
    nb = prod(query.shape[:-3])
    lse = torch.empty((nb, nh, nq), dtype=torch.float32, device=query.device)
    evoformer_attention_fwd(query, key, value, bias1, bias2, out, lse)
    return out, lse


def _attention_bwd(
    out_grad: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
    bias1_grad: torch.Tensor,
    bias2_grad: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert max(query.shape[-1], value.shape[-1]) <= 64, "Hidden dimension is too large"
    dq = torch.empty_like(query, dtyp=query.dtype)
    dk = torch.empty_like(key, dtyp=key.dtype)
    dv = torch.empty_like(value, dtyp=value.dtype)
    delta = torch.empty_like(lse)
    if bias1_grad:
        db1 = torch.zeros_like(bias1, dtype=torch.float32)
    else:
        db1 = torch.tensor([], dtype=torch.float32, device=bias1.device)
    if bias2_grad:
        db2 = torch.zeros_like(bias2, dtype=torch.float32)
    else:
        db2 = torch.tensor([], dtype=torch.float32, device=bias2.device)
    evoformer_attention_bwd(out_grad, query, key, value, out, lse, delta, bias1, bias2, dq, dk, dv, db1, db2)
    return dq, dk, dv, db1.to(out_grad.dtype), db2.to(out_grad.dtype)


class EvoformerFusedAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias1: Optional[torch.Tensor] = None,
        bias2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bias1 = bias1.contiguous() if bias1 is not None else torch.tensor([], dtype=query.dtype, device=query.device)
        bias2 = bias2.contiguous() if bias2 is not None else torch.tensor([], dtype=query.dtype, device=query.device)
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        out, lse = _attention_fwd(query, key, value, bias1, bias2)
        ctx.save_for_backward(query, key, value, out, lse, bias1, bias2)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, out, lse, bias1, bias2 = ctx.saved_tensors
        is_b1_grad = bias1.numel() != 0 and ctx.needs_input_grad[3]
        is_b2_grad = bias2.numel() != 0 and ctx.needs_input_grad[4]
        dq, dk, dv, db1, db2 = _attention_bwd(
            grad_output,
            query,
            key,
            value,
            out,
            lse,
            bias1,
            bias2,
            is_b1_grad,
            is_b2_grad,
        )
        if not is_b1_grad:
            db1 = None
        if not is_b2_grad:
            db2 = None
        return dq, dk, dv, db1, db2


def _evoformer_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: List[torch.Tensor],
) -> torch.Tensor:

    assert len(biases) <= 2

    if len(biases) in (0, 1):
        biases.append(None)

    bias_1_shape = lambda x: (x.shape[0], x.shape[1], 1, 1, x.shape[2])
    bias_2_shape = lambda x: (x.shape[0], 1, x.shape[3], x.shape[2], x.shape[2])

    if biases[0] is not None:
        assert biases[0].shape == bias_1_shape(query), "The first bias has incorrect shape"

    if biases[1] is not None:
        assert biases[1].shape == bias_2_shape(query), "The second bias has incorrect shape"

    return EvoformerFusedAttention.apply(query, key, value, biases[0], biases[1])


@torch.jit.ignore
def evoformer_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: List[torch.Tensor],
) -> torch.Tensor:
    r"""
    Compute attention using the DeepSpeed's evoformer kernel.

    Args:
        query:
            [*, H, Q, C] query data
        key:
            [*, H, K, C] key data
        value:
            [*, H, V, C] value data
        biases (List[torch.Tensor]):
            List of biases that broadcast to [*, H, Q, K]

    Note:
        Assert that all input tensors have the same dtype.
    """

    def reshape_dim(x: torch.Tensor) -> torch.Tensor:
        num_batch_dims = len(x.shape[:-3])
        if num_batch_dims < 2:
            return x.reshape(*((1,) * (2 - num_batch_dims) + x.shape))
        if num_batch_dims > 2:
            return x.reshape(*((x.shape[0], -1) + x.shape[-3:]))
        return x

    # [*, Q/K, H, C]
    q = query.transpose(-2, -3)
    k = key.transpose(-2, -3)
    v = value.transpose(-2, -3)

    # Reshape tensors to match with expected input shape [B, N, Q/K, H, C]
    orig_shape = q.shape
    if len(orig_shape[:-3]) != 2:
        q = reshape_dim(q)
        k = reshape_dim(k)
        v = reshape_dim(v)
        bx = [reshape_dim(b) for b in biases]

    # The kernel requires inputs to be type bf16 or fp16
    orig_dtype = q.dtype
    if orig_shape not in (torch.bfloat16, torch.float16):
        o = _evoformer_attention(
            q.to(dtype=torch.float16),
            k.to(dtype=torch.float16),
            v.to(dtype=torch.float16),
            [b.to(dtype=torch.float16) for b in bx],
        ).to(dtype=orig_dtype)
    else:
        o = _evoformer_attention(q, k, v, bx)

    # Reshape to match with the input shape
    o = o.reshape(orig_shape)

    return o
