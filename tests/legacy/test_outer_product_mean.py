import unittest

import haiku as hk
import jax
import numpy as np
import torch

from deepfold.model.alphafold.nn.outer_product_mean import ParallelOuterProductMean as OuterProductMean
from deepfold.model.alphafold.utils import import_weights
from deepfold.utils.tensor_utils import tree_map
from tests import compare_utils
from tests.alphafold_model import modules
from .config import consts
from .init import init_parallel


class TestOuterProductMean(unittest.TestCase):
    def test_shape(self):
        c = 32

        init_parallel(local_rank=0, world_size=1)
        opm = OuterProductMean(consts.c_m, consts.c_z, c)

        m = torch.rand((consts.batch_size, consts.n_seq, consts.n_res, consts.c_m))
        mask = torch.randint(0, 2, size=(consts.batch_size, consts.n_seq, consts.n_res))
        m = opm(m, mask=mask, chunk_size=None)

        self.assertTrue(m.shape == (consts.batch_size, consts.n_res, consts.n_res, consts.c_z))

    def test_opm_compare(self):
        def run_opm(msa_act, msa_mask):
            config = compare_utils.get_alphafold_config()
            c_evo = config.model.embeddings_and_evoformer.evoformer
            opm = modules.OuterProductMean(
                c_evo.outer_product_mean,
                config.model.global_config,
                consts.c_z,
            )
            act = opm(act=msa_act, mask=msa_mask)
            return act

        f = hk.transform(run_opm)

        n_res = consts.n_res
        n_seq = consts.n_seq
        c_m = consts.c_m

        opm_path = "alphafold/alphafold_iteration/evoformer/evoformer_iteration/outer_product_mean"

        msa_act = np.random.rand(n_seq, n_res, c_m).astype(np.float32) * 100
        msa_mask = np.random.randint(low=0, high=2, size=(n_seq, n_res)).astype(np.float32)

        # Fetch pretrained parameters (but only from one block)]
        params = compare_utils.fetch_alphafold_module_weights_to_haiku(opm_path)
        params = tree_map(lambda a: a[0], params, jax.Array)

        out_gt = f.apply(params, None, msa_act, msa_mask).block_until_ready()
        out_gt = torch.as_tensor(np.array(out_gt))

        # Prepare MegaFold
        init_parallel(local_rank=0, world_size=1)
        opm = OuterProductMean(consts.c_m, consts.c_z, 32, eps=1e-3).cuda()

        params = compare_utils.fetch_alphafold_module_weights_to_dict(opm_path)
        params = tree_map(lambda a: torch.tensor(a[0]).cuda(), params, np.ndarray)
        trans_dict = {"/outer_product_mean": import_weights.OuterProductMeanParams(opm)}
        trans_dict = import_weights.process_translation_dict(trans_dict, top_layer=False)
        import_weights.assign(trans_dict, params)

        out_repro = opm(
            torch.as_tensor(msa_act).cuda(),
            mask=torch.as_tensor(msa_mask).cuda(),
            chunk_size=32,
        ).cpu()

        # Even when correct, OPM has large, precision-related errors
        self.assertTrue(torch.max(torch.abs(out_gt - out_repro)) < 5e-4)


if __name__ == "__main__":
    unittest.main()
