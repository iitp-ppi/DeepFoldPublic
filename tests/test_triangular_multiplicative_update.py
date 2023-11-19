import unittest

import haiku as hk
import jax
import numpy as np
import torch

from deepfold.model.alphafold.nn.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from deepfold.model.alphafold.utils import import_weights
from deepfold.utils.tensor_utils import tree_map
from tests import compare_utils
from tests.alphafold_model import modules
from tests.config import consts
from tests.init import init_parallel


class TestTriangularMultiplicativeUpdate(unittest.TestCase):
    def test_shape(self):
        c_z = consts.c_z
        c = 11

        init_parallel(local_rank=0, world_size=1)
        tm = TriangleMultiplicationOutgoing(c_z, c)

        n_res = consts.c_z
        batch_size = consts.batch_size

        x = torch.rand((batch_size, n_res, n_res, c_z))
        mask = torch.randint(0, 2, size=(batch_size, n_res, n_res))
        shape_before = x.shape
        x = tm(x, mask)
        shape_after = x.shape

        self.assertTrue(shape_before == shape_after)

    def _tri_mul_compare(self, incoming=False, chunk_size=None):
        name = "triangle_multiplication_" + ("incoming" if incoming else "outgoing")

        def run_tri_mul(pair_act, pair_mask):
            config = compare_utils.get_alphafold_config()
            c_e = config.model.embeddings_and_evoformer.evoformer
            tri_mul = modules.TriangleMultiplication(
                c_e.triangle_multiplication_incoming if incoming else c_e.triangle_multiplication_outgoing,
                config.model.global_config,
                name=name,
            )
            act = tri_mul(left_act=pair_act, left_mask=pair_mask)
            return act

        f = hk.transform(run_tri_mul)

        n_res = consts.n_res

        weight_path = f"alphafold/alphafold_iteration/evoformer/evoformer_iteration/{name}"

        pair_act = np.random.rand(n_res, n_res, consts.c_z).astype(np.float32)
        pair_mask = np.random.randint(low=0, high=2, size=(n_res, n_res))
        pair_mask = pair_mask.astype(np.float32)

        # Fetch pretrained parameters (but only from one block)]
        params = compare_utils.fetch_alphafold_module_weights_to_haiku(weight_path)
        params = tree_map(lambda a: a[0], params, jax.Array)

        out_gt = out_gt = f.apply(params, None, pair_act, pair_mask).block_until_ready()
        out_gt = torch.as_tensor(np.array(out_gt))

        # Prepare MegaFold
        init_parallel(local_rank=0, world_size=1)
        if incoming:
            tri_mul = TriangleMultiplicationIncoming(consts.c_z, c_hidden=128).cuda()
        else:
            tri_mul = TriangleMultiplicationOutgoing(consts.c_z, c_hidden=128).cuda()

        params = compare_utils.fetch_alphafold_module_weights_to_dict(weight_path)
        params = tree_map(lambda a: torch.tensor(a[0]).cuda(), params, np.ndarray)
        if incoming:
            trans_dict = {"/triangle_multiplication_incoming": import_weights.TriMulInParams(tri_mul)}
        else:
            trans_dict = {"/triangle_multiplication_outgoing": import_weights.TriMulOutParams(tri_mul)}
        trans_dict = import_weights.process_translation_dict(trans_dict, top_layer=False)

        import_weights.assign(trans_dict, params)

        out_repro = tri_mul(
            torch.as_tensor(pair_act, dtype=torch.float32).cuda(),
            mask=torch.as_tensor(pair_mask, dtype=torch.float32).cuda(),
            chunk_size=chunk_size,
        ).cpu()

        self.assertTrue(torch.max(torch.abs(out_gt - out_repro)) < consts.eps * 0.5)

    def test_tri_mul_out_compare(self):
        self._tri_mul_compare(incoming=False)

    def test_tri_mul_in_compare(self):
        self._tri_mul_compare(incoming=True)

    def test_tri_mul_out_chunk(self):
        self._tri_mul_compare(incoming=False, chunk_size=4)

    def test_tri_mul_in_chunk(self):
        self._tri_mul_compare(incoming=True, chunk_size=4)


if __name__ == "__main__":
    unittest.main()
