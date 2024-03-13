import unittest
from pathlib import Path

import numpy as np
import torch

from deepfold.config import AlphaFoldConfig
from deepfold.import_utils import import_jax_weights_
from deepfold.modules.alphafold import AlphaFold


class TestImportWeights(unittest.TestCase):
    def test_import_jax_weights_(self):
        npz_path = Path("/scratch/alphafold.data/params/params_model_1_ptm.npz")

        config = AlphaFoldConfig.from_preset(
            is_multimer=False,
            enable_ptm=True,
            enable_templates=True,
        )
        model = AlphaFold(config=config)
        model.eval()

        import_jax_weights_(
            model=model,
            npz_path=str(npz_path),
            is_multimer=False,
            enable_ptm=True,
            enable_templates=True,
            fuse_projection_weights=False,
        )

        data = np.load(npz_path)
        prefix = "alphafold/alphafold_iteration/"

        test_pairs = [
            # Normal linear weight
            (
                torch.as_tensor(data[prefix + "structure_module/initial_projection//weights"]).transpose(-1, -2),
                model.structure_module.linear_in.weight,
            ),
            # Normal layer norm param
            (
                torch.as_tensor(
                    data[prefix + "evoformer/prev_pair_norm//offset"],
                ),
                model.recycling_embedder.layer_norm_z.bias,
            ),
            # From a stack
            (
                torch.as_tensor(
                    data[prefix + ("evoformer/evoformer_iteration/outer_product_mean/" "left_projection//weights")][
                        1
                    ].transpose(-1, -2)
                ),
                model.evoformer_stack.blocks[1].outer_product_mean.linear_1.weight,
            ),
        ]

        for w_alpha, w_repro in test_pairs:
            self.assertTrue(torch.all(w_alpha == w_repro))


if __name__ == "__main__":
    unittest.main()
