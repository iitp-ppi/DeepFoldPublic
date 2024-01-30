import sys
import unittest

import torch
import torch.nn as nn

from deepfold.model.alphafold.nn.primitives import Linear, _attention

torch.set_printoptions(linewidth=120, precision=3, threshold=sys.maxsize)


class TestPrimitives(unittest.TestCase):
    def test_linear(self):
        c_in = 3
        c_out = 3
        seq = 5

        with torch.no_grad():
            df_linear = Linear(c_in, c_out).cuda()
            linear = nn.Linear(c_in, c_out).cuda()

            df_linear.weight.copy_(linear.weight)
            df_linear.bias.copy_(linear.bias)

            x = torch.randn((seq, c_in)).cuda()

            out_1 = df_linear(x)
            out_2 = linear(x)

        self.assertTrue(torch.allclose(out_1, out_2))

    def test_attention_core_forward(self):
        n_seq = 8
        n_res = 16
        n_heads = 4
        c = 128
        dtype = torch.float32
        eps = 1e-6
        inf = 1e9

        q = torch.rand([n_seq, n_heads, n_res, c], dtype=dtype).cuda()
        k = torch.rand([n_seq, n_heads, n_res, c], dtype=dtype).cuda()
        v = torch.rand([n_seq, n_heads, n_res, c], dtype=dtype).cuda()
        mask = torch.randint(0, 2, [n_seq, n_res]).cuda()
        mask_bias = (inf * mask - 1)[..., None, None, :].to(dtype)

        out_1 = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask_bias, scale=1.0)
        out_2 = _attention(q, k, v, [mask_bias])

        self.assertTrue(torch.allclose(out_1, out_2))


if __name__ == "__main__":
    unittest.main()
