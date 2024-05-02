import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

import deepfold.modules.inductor as inductor
from deepfold.common import protein
from deepfold.config import AlphaFoldConfig, FeaturePipelineConfig
from deepfold.data import feature_pipeline
from deepfold.modules.alphafold import AlphaFold
from deepfold.runner.plot import plot_distogram, plot_msa, plot_plddt, plot_predicted_alignment_error
from deepfold.runner.pseudo_3d import plot_protein
from deepfold.runner.utils import TqdmHandler
from deepfold.utils.file_utils import dump_pickle, load_pickle
from deepfold.utils.import_utils import import_jax_weights_
from deepfold.utils.tensor_utils import tensor_tree_map

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path, mode: str = "a") -> None:
    log_file.parent.mkdir(exist_ok=True, parents=True)
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            handler.close()
            root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[TqdmHandler(), logging.FileHandler(log_file, mode=mode)],
        force=True,
    )


def main(target_name: str, npz_path: str, random_seed: int = 0, device: str = "cuda"):
    inductor.disable()
    device = torch.device(device)

    # Random
    if random_seed is None:
        random_seed = random.randrange(2**32)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    model_name = os.path.split(os.path.splitext(npz_path)[0])[-1]
    is_multimer = "multimer" in model_name
    prefix = model_name

    # Move working directory
    workdir = Path(os.path.join("_runs", target_name))

    if is_multimer:
        feature_dict = load_pickle(workdir / "features.pkl")
    else:
        feature_dict = load_pickle(workdir / "features.pkl")

    # Move working directory
    workdir = Path(os.path.join("_runs", target_name))

    # Setup logging
    setup_logging(log_file=(workdir / "log"))

    logger.info(f"Set seed to {random_seed}")
    logger.info(f"Save outputs to '{workdir}'")

    plot_msa(feature_dict).savefig(workdir / f"{prefix}_seed_{random_seed}_msa.png")
    plt.close()

    feature_config = FeaturePipelineConfig.from_preset(
        preset="predict",
        is_multimer=is_multimer,
        ensemble_seed=random_seed,
        # max_recycling_iters=3,  # NOTE: Early stop for debugging
    )

    processed_feature_dict = feature_pipeline.FeaturePipeline(config=feature_config).process_features(feature_dict)

    processed_feature_dict = {
        k: torch.as_tensor(v[None, ...]).to(device=device) for k, v in processed_feature_dict.items()
    }  # Batch dimension is required

    logger.info(f"seqlen={processed_feature_dict['aatype'].shape[1]}")  # [batch, N_res, N_recycle]

    enable_ptm = "ptm" in npz_path or is_multimer
    enable_templates = not any(model_name.endswith(x) for x in ["_3", "_4", "_5"])
    fuse_projection_weights = model_name.endswith("multimer_v3")

    if not enable_templates:
        logger.info("Disable templates...")

    model_config = AlphaFoldConfig.from_preset(
        is_multimer=is_multimer,
        enable_ptm=enable_ptm,
        enable_templates=enable_templates,
        inference_chunk_size=4,
        inference_block_size=None,  # 256
    )

    logger.info(f"Load the model '{model_name}'")
    model = AlphaFold(config=model_config).to(device=device)
    model.eval()
    import_jax_weights_(
        model=model,
        npz_path=npz_path,
        is_multimer=is_multimer,
        enable_ptm=enable_ptm,
        enable_templates=enable_templates,
        fuse_projection_weights=fuse_projection_weights,
    )

    logger.info("Start inference procedure")
    time_begin = time.time()
    torch.cuda.reset_peak_memory_stats()
    out = model(processed_feature_dict, save_trajectory=True, verbose=True)
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

    logger.info("Save outputs")
    dump_pickle(processed_feature_dict, workdir / f"{prefix}_seed_{random_seed}_processed.pkl")
    dump_pickle(out, workdir / f"{prefix}_seed_{random_seed}_output.pkl")

    prots = protein.from_prediction(
        processed_features=processed_feature_dict,
        result=out,
        b_factors=out["plddt"],
        remark=prefix,
        trajectory=True,
    )
    with open(workdir / f"{prefix}_seed_{random_seed}_unrelaxed.pdb", "w") as fp:
        fp.write(protein.to_pdb(prots))

    logger.info("Plot figures")
    outputs = {"model_1_multimer_v3": out}
    asym_id = processed_feature_dict.get("asym_id", None)
    plot_plddt(outputs, asym_id=asym_id).savefig(workdir / f"{prefix}_seed_{random_seed}_plddt.png")
    plt.close()
    if enable_ptm:
        plot_predicted_alignment_error(outputs, asym_id=asym_id).savefig(workdir / f"{prefix}_seed_{random_seed}_pae.png")
        plt.close()
    plot_distogram(outputs, asym_id=asym_id).savefig(workdir / f"{prefix}_seed_{random_seed}_distogram.png")
    plt.close()
    plot_protein(protein=prots[-1]).savefig(workdir / f"{prefix}_seed_{random_seed}_plot.png", dpi=300)
    plt.close()

    del model


if __name__ == "__main__":
    # main("T1113o", is_multimer=True)
    # main("H1106", is_multimer=True)
    # main("T1006")
    # main("T1109")
    npz_paths = [
        "/gpfs/database/casp16/af_params/params_model_1.npz",
        "/gpfs/database/casp16/af_params/params_model_2.npz",
        "/gpfs/database/casp16/af_params/params_model_3.npz",
        "/gpfs/database/casp16/af_params/params_model_4.npz",
        "/gpfs/database/casp16/af_params/params_model_5.npz",
        "/runs/database/params/deepfold_v1/deepfold_model_1.npz",
        "/runs/database/params/deepfold_v1/deepfold_model_2.npz",
        "/runs/database/params/deepfold_v1/deepfold_model_3.npz",
        "/runs/database/params/deepfold_v1/deepfold_model_4.npz",
        "/runs/database/params/deepfold_v1/deepfold_model_5.npz",
    ]
    # main("T1201")
    # npz_paths = [
    #     "/gpfs/database/casp16/af_params/params_model_1_multimer_v2.npz",
    #     "/gpfs/database/casp16/af_params/params_model_2_multimer_v2.npz",
    #     "/gpfs/database/casp16/af_params/params_model_3_multimer_v2.npz",
    #     "/gpfs/database/casp16/af_params/params_model_4_multimer_v2.npz",
    #     "/gpfs/database/casp16/af_params/params_model_5_multimer_v2.npz",
    # ]
    npz_paths = [
        "/gpfs/database/casp16/af_params/params_model_1_multimer.npz",
        "/gpfs/database/casp16/af_params/params_model_2_multimer.npz",
        "/gpfs/database/casp16/af_params/params_model_3_multimer.npz",
        "/gpfs/database/casp16/af_params/params_model_4_multimer.npz",
        "/gpfs/database/casp16/af_params/params_model_5_multimer.npz",
        "/gpfs/database/casp16/af_params/params_model_1_multimer_v2.npz",
        "/gpfs/database/casp16/af_params/params_model_2_multimer_v2.npz",
        "/gpfs/database/casp16/af_params/params_model_3_multimer_v2.npz",
        "/gpfs/database/casp16/af_params/params_model_4_multimer_v2.npz",
        "/gpfs/database/casp16/af_params/params_model_5_multimer_v2.npz",
        "/gpfs/database/casp16/af_params/params_model_1_multimer_v3.npz",
        "/gpfs/database/casp16/af_params/params_model_2_multimer_v3.npz",
        "/gpfs/database/casp16/af_params/params_model_3_multimer_v3.npz",
        "/gpfs/database/casp16/af_params/params_model_4_multimer_v3.npz",
        "/gpfs/database/casp16/af_params/params_model_5_multimer_v3.npz",
    ]
    for npz_path in npz_paths:
        main("H1134", npz_path=npz_path, random_seed=1398, device="cuda:1")
