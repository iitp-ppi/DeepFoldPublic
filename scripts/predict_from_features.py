import logging
import random
import time

import numpy as np
import torch

import deepfold.modules.inductor as inductor
from deepfold.common import protein
from deepfold.config import AlphaFoldConfig, FeaturePipelineConfig
from deepfold.data import feature_pipeline
from deepfold.import_utils import import_jax_weights_
from deepfold.modules.alphafold import AlphaFold
from deepfold.utils.file_utils import dump_pickle, load_pickle
from deepfold.utils.tensor_utils import tensor_tree_map

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def main():
    is_multimer = True
    inductor.enable()

    device = torch.device("cuda")
    # Random
    random_seed = 1398
    if random_seed is None:
        random_seed = random.randrange(2**32)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    if is_multimer:
        feature_dict = load_pickle("_runs/T1109/features_multimer.pkl")
    else:
        feature_dict = load_pickle("_runs/T1109/features.pkl")
    print("=== Feature Dict ===")
    for k, v in feature_dict.items():
        print(f"{k} :", tuple(v.shape))
    print()

    feature_config = FeaturePipelineConfig.from_preset(
        preset="predict",
        is_multimer=is_multimer,
        ensemble_seed=random_seed,
    )
    print("=== Pipeline Config ===")
    for k, v in feature_config.to_dict().items():
        print(f"{k} :", v)
    print()

    processed_feature_dict = feature_pipeline.FeaturePipeline(config=feature_config).process_features(feature_dict)
    print("=== Batch ===")
    for k, v in processed_feature_dict.items():
        print(f"{k} :", tuple(v.shape))
    print()

    processed_feature_dict = {
        k: torch.as_tensor(v[None, ...]).to(device=device) for k, v in processed_feature_dict.items()
    }  # Batch dimension is required

    if is_multimer:
        npz_path = "/scratch/alphafold.data/params/params_model_1_multimer_v3.npz"
    else:
        npz_path = "/scratch/alphafold.data/params/params_model_1_ptm.npz"

    model_config = AlphaFoldConfig.from_preset(
        is_multimer=is_multimer,
        enable_ptm=True,
        enable_templates=True,
        # inference_chunk_size=None,
    )
    print("=== Model Config ===")
    for k, v in model_config.to_dict().items():
        print(f"{k} :", v)
    print()

    model = AlphaFold(config=model_config).to(device=device)
    model.eval()
    import_jax_weights_(
        model=model,
        npz_path=str(npz_path),
        is_multimer=is_multimer,
        enable_ptm=True,
        enable_templates=True,
        fuse_projection_weights=is_multimer,
    )

    time_begin = time.time()
    torch.cuda.reset_peak_memory_stats()
    out = model(processed_feature_dict)
    time_elapsed = time.time() - time_begin

    logger.info(f"Time elapsed: {time_elapsed:.2f} sec")
    logger.info(f"CUDA max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")

    # Remove batch diemnsion
    processed_feature_dict = tensor_tree_map(
        fn=lambda x: np.array(x[..., -1].squeeze(0).cpu()),
        tree=processed_feature_dict,
    )

    # Remove batch diemnsion
    out = tensor_tree_map(
        fn=lambda x: np.array(x.squeeze(0).cpu()),
        tree=out,
    )

    dump_pickle(processed_feature_dict, "processed.pkl")
    dump_pickle(out, "output.pkl")

    unrelaxed_protein = protein.from_prediction(
        processed_features=processed_feature_dict,
        result=out,
        b_factors=out["plddt"],
        remove_leading_feature_dimension=False,
        remark="T1109",
    )

    with open("unrelaxed.pdb", "w") as fp:
        fp.write(protein.to_pdb(unrelaxed_protein))


if __name__ == "__main__":
    main()
