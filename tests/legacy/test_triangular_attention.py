import unittest

import haiku as hk
import jax
import numpy as np
import torch

from deepfold.model.alphafold.nn.triangular_attention import TriangleAttention
from deepfold.model.alphafold.utils import import_weights
from deepfold.utils.tensor_utils import tree_map
from tests import compare_utils
from tests.alphafold_model import modules

from .config import consts
from .init import init_parallel


class TestTriangularAttention(unittest.TestCase):
    def test_shape(self):
        c_z = consts.c_z
        c = 12
        no_heads = 4
        starting = True

        init_parallel(local_rank=0, world_size=1)
        tan = TriangleAttention(c_z, c, no_heads, starting=starting)

        batch_size = consts.batch_size
        n_res = consts.n_res

        x = torch.rand((batch_size, n_res, n_res, c_z))
        shape_before = x.shape
        x = tan(x, chunk_size=None)
        shape_after = x.shape

        self.assertTrue(shape_before == shape_after)

    def _tri_att_compare(self, starting=False, chunk_size=None):
        name = "triangle_attention_" + ("starting" if starting else "ending") + "_node"

        def run_tri_att(pair_act, pair_mask):
            config = compare_utils.get_alphafold_config()
            c_e = config.model.embeddings_and_evoformer.evoformer
            tri_att = modules.TriangleAttention(
                c_e.triangle_attention_starting_node if starting else c_e.triangle_attention_ending_node,
                config.model.global_config,
                name=name,
            )
            act = tri_att(pair_act=pair_act, pair_mask=pair_mask)
            return act

        f = hk.transform(run_tri_att)

        n_res = consts.n_res

        weight_path = f"alphafold/alphafold_iteration/evoformer/evoformer_iteration/{name}"

        pair_act = np.random.rand(n_res, n_res, consts.c_z) * 100
        pair_mask = np.random.randint(low=0, high=2, size=(n_res, n_res))

        # Fetch pretrained parameters (but only from one block)]
        params = compare_utils.fetch_alphafold_module_weights_to_haiku(weight_path)
        params = tree_map(lambda a: a[0], params, jax.Array)

        out_gt = f.apply(params, None, pair_act, pair_mask).block_until_ready()
        out_gt = torch.as_tensor(np.array(out_gt))

        # Prepare MegaFold
        init_parallel(local_rank=0, world_size=1)
        msa_att = TriangleAttention(consts.c_z, c_hidden=32, num_heads=4, starting=starting).cuda()

        params = compare_utils.fetch_alphafold_module_weights_to_dict(weight_path)
        params = tree_map(lambda a: torch.tensor(a[0]).cuda(), params, np.ndarray)
        if starting:
            trans_dict = {"/triangle_attention_starting_node": import_weights.TriAttParams(msa_att)}
        else:
            trans_dict = {"/triangle_attention_ending_node": import_weights.TriAttParams(msa_att)}
        trans_dict = import_weights.process_translation_dict(trans_dict, top_layer=False)

        import_weights.assign(trans_dict, params)

        out_repro = msa_att(
            torch.as_tensor(pair_act, dtype=torch.float32).cuda(),
            mask=torch.as_tensor(pair_mask, dtype=torch.float32).cuda(),
            chunk_size=chunk_size,
        ).cpu()

        self.assertTrue(torch.max(torch.abs(out_gt - out_repro)) < consts.eps)

    def test_tri_att_end_compare(self):
        self._tri_att_compare(starting=False)

    def test_tri_att_start_compare(self):
        self._tri_att_compare(starting=True)

    def test_tri_att_end_chunk(self):
        self._tri_att_compare(starting=False, chunk_size=32)

    def test_tri_att_start_chunk(self):
        self._tri_att_compare(starting=True, chunk_size=32)


if __name__ == "__main__":
    unittest.main()
