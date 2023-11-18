import unittest

import haiku as hk
import jax
import numpy as np
import torch

from deepfold.model.alphafold.nn.msa import MSAColumnAttention, MSAColumnGlobalAttention, MSARowAttentionWithPairBias
from deepfold.model.alphafold.utils import import_weights
from deepfold.utils.tensor_utils import tree_map
from tests import compare_utils
from tests.alphafold_model import modules
from tests.config import consts
from tests.init import init_parallel


class TestMSARowAttentionWithPairBias(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_seq = consts.n_seq
        n_res = consts.n_res
        c_m = consts.c_m
        c_z = consts.c_z
        c = 52
        no_heads = 4
        chunk_size = None

        init_parallel(local_rank=0, world_size=1)
        mrapb = MSARowAttentionWithPairBias(c_m, c_z, c, no_heads)

        m = torch.rand((batch_size, n_seq, n_res, c_m))
        z = torch.rand((batch_size, n_res, n_res, c_z))

        shape_before = m.shape
        m = mrapb(m, z=z, chunk_size=chunk_size)
        shape_after = m.shape

        self.assertTrue(shape_before == shape_after)

    def test_compare(self):
        def run_msa_row_att(msa_act, msa_mask, pair_act):
            config = compare_utils.get_alphafold_config()
            c_e = config.model.embeddings_and_evoformer.evoformer
            msa_row = modules.MSARowAttentionWithPairBias(
                c_e.msa_row_attention_with_pair_bias,
                config.model.global_config,
            )
            act = msa_row(msa_act=msa_act, msa_mask=msa_mask, pair_act=pair_act)
            return act

        f = hk.transform(run_msa_row_att)

        n_res = consts.n_res
        n_seq = consts.n_seq

        weight_path = "alphafold/alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention"

        msa_act = np.random.rand(n_seq, n_res, consts.c_m).astype(np.float32)
        msa_mask = np.random.randint(low=0, high=2, size=(n_seq, n_res)).astype(np.float32)
        pair_act = np.random.rand(n_res, n_res, consts.c_z).astype(np.float32)

        # Fetch pretrained parameters (but only from one block)]
        params = compare_utils.fetch_alphafold_module_weights_to_haiku(weight_path)
        params = tree_map(lambda a: a[0], params, jax.Array)

        out_gt = f.apply(params, None, msa_act, msa_mask, pair_act).block_until_ready()
        out_gt = torch.as_tensor(np.array(out_gt))

        # Prepare MegaFold
        init_parallel(local_rank=0, world_size=1)
        msa_att = MSARowAttentionWithPairBias(consts.c_m, consts.c_z, c_hidden=32, num_heads=8).cuda()

        params = compare_utils.fetch_alphafold_module_weights_to_dict(weight_path)
        params = tree_map(lambda a: torch.tensor(a[0]).cuda(), params, np.ndarray)
        trans_dict = {"/msa_row_attention_with_pair_bias": import_weights.MSAAttPairBiasParams(msa_att)}
        trans_dict = import_weights.process_translation_dict(trans_dict, top_layer=False)

        import_weights.assign(trans_dict, params)

        out_repro = msa_att(
            torch.as_tensor(msa_act).cuda(),
            torch.as_tensor(pair_act).cuda(),
            mask=torch.as_tensor(msa_mask).cuda(),
            chunk_size=32,
        ).cpu()

        # Even when correct, OPM has large, precision-related errors
        self.assertTrue(torch.max(torch.abs(out_gt - out_repro)) < consts.eps)


# TODO: MSACol
# TODO: MSAGlobal

if __name__ == "__main__":
    unittest.main()
