import unittest
from typing import List

import torch
import torch.nn.functional as F

from deepfold.core.ops.evoformer_attention import evoformer_attention_core


def attention_reference(
    q_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
    k_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
    v_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
    biases: List[torch.Tensor],
    sm_scale: float,
) -> torch.Tensor:
    q = q_input.transpose(-2, -3)
    k = k_input.transpose(-2, -3)
    v = v_input.transpose(-2, -3)
    k_t = k.transpose(-1, -2)
    a = torch.matmul(q, k_t) * sm_scale

    for b in biases:
        a += b

    a = F.softmax(a, dim=-1)
    a_v = torch.matmul(a, v)
    o = a_v.transpose(-2, -3)

    return o


class EvoformerAttention(unittest.TestCase):
    def test_EvoformerAttention(self):
        device = torch.device("cuda")
        for dtype in [torch.float16, torch.bfloat16]:
            for tensor_shape in [(1, 255, 255, 4, 16), (1, 512, 256, 8, 8)]:
                batch, n, seq_len, heads, dim = tensor_shape

                Q = torch.randn(batch, n, seq_len, heads, dim, dtype=dtype, device=device, requires_grad=True)
                K = torch.randn(batch, n, seq_len, heads, dim, dtype=dtype, device=device, requires_grad=True)
                V = torch.randn(batch, n, seq_len, heads, dim, dtype=dtype, device=device, requires_grad=True)
                mask = torch.randint(0, 2, (batch, n, 1, 1, seq_len), dtype=dtype, device=device)
                mask_bias = 1e9 * (mask - 1)
                bias = torch.randn(
                    batch,
                    1,
                    heads,
                    seq_len,
                    seq_len,
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                )
                dummy_out = torch.rand_like(Q, dtype=dtype, device=device)
                ref_out = attention_reference(Q, K, V, [mask_bias, bias], 1 / (dim**0.5))
                ref_out.backward(dummy_out)
                ref_dv, V.grad = V.grad.clone(), None
                ref_dk, K.grad = K.grad.clone(), None
                ref_dq, Q.grad = Q.grad.clone(), None
                ref_db, bias.grad = bias.grad.clone(), None

                out = evoformer_attention_core(Q, K, V, [mask_bias, bias])
                out.backward(dummy_out)
                dv, v_grad = V.grad.clone(), None
                dk, k_grad = K.grad.clone(), None
                dq, q_grad = Q.grad.clone(), None
                db, bias.grad = bias.grad.clone(), None

                eps = 1e-2 if dtype == torch.float16 else 5e-2

                self.assertTrue(
                    torch.max(torch.abs(ref_out - out)).item() < eps,
                    f"out {dtype} eps: {torch.max(torch.abs(ref_out - out))}",
                )
                self.assertTrue(
                    torch.max(torch.abs(ref_dv - dv)) < eps,
                    f"dv {dtype} eps: {torch.max(torch.abs(ref_dv - dv))}",
                )
                self.assertTrue(
                    torch.max(torch.abs(ref_dk - dk)) < eps,
                    f"dk {dtype} eps: {torch.max(torch.abs(ref_dk - dk))}",
                )
                self.assertTrue(
                    torch.max(torch.abs(ref_dq - dq)) < eps,
                    f"dq {dtype} eps: {torch.max(torch.abs(ref_dq - dq))}",
                )
                self.assertTrue(
                    torch.max(torch.abs(ref_db - db)) < 2 * eps,
                    f"db {dtype} eps: {torch.max(torch.abs(ref_db - db))}",
                )
