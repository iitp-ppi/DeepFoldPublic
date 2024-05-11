import argparse
import logging
import os
import random
import signal
import time
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed

import deepfold.distributed as dist
import deepfold.distributed.model_parallel as mp
import deepfold.modules.inductor as inductor
from deepfold.common import protein
from deepfold.config import AlphaFoldConfig, FeaturePipelineConfig
from deepfold.data import feature_pipeline
from deepfold.modules.alphafold import AlphaFold
from deepfold.utils.file_utils import dump_pickle, load_pickle
from deepfold.utils.import_utils import import_jax_weights_
from deepfold.utils.iter_utils import flatten_dict
from deepfold.utils.random import NUMPY_SEED_MODULUS
from deepfold.utils.tensor_utils import tensor_tree_map

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_features_filepath",
        type=Path,
        required=True,
        help="Path to input feature file generated by sequence alignment processing.",
    )
    parser.add_argument(
        "--output_dirpath",
        type=Path,
        required=True,
        help="Path to prediction output directory.",
    )
    parser.add_argument(
        "--params_dirpath",
        type=Path,
        required=True,
        help="Path to parameter NPZ file.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        help="Model preset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Global seed for pseudo-random number generators.",
    )
    parser.add_argument(
        "--mp_size",
        type=int,
        default=0,
        help="Model parallelism (MP) size. Set 0 to disable mp.",
    )
    parser.add_argument(
        "--save_recycle",
        action="store_true",
        help="Whether to save a recycling outputs.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to output files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite.",
    )
    parser.add_argument(
        "--use_multimer_templates",
        default="",
        type=str,
        dest="multimer_templates",
        help="Enable multimer templates.",
    )
    args = parser.parse_args()
    #
    if args.mp_size != 0:
        assert torch.cuda.device_count() == args.mp_size
    #
    return args


def create_alphafold_module(
    alphafold_config: AlphaFoldConfig,
    device: torch.device,
    seed: int,
) -> AlphaFold:
    numpy_random_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    torch_cuda_rng_state = torch.cuda.get_rng_state(device=device)
    np.random.seed(seed % NUMPY_SEED_MODULUS)
    torch.manual_seed(seed)
    alphafold = AlphaFold(config=alphafold_config)
    alphafold.to(device=device)
    torch.cuda.set_rng_state(torch_cuda_rng_state, device=device)
    torch.set_rng_state(torch_rng_state)
    np.random.set_state(numpy_random_state)
    return alphafold


def get_preset_opts(preset: str) -> Tuple[str, Tuple[dict, dict, dict]]:
    is_multimer = "multimer" in preset
    enable_ptm = "ptm" in preset or is_multimer
    enable_templates = not any(preset.endswith(x) for x in ["_3", "_4", "_5"])
    fuse_projection_weights = preset.endswith("multimer_v3")

    model_cfg_kwargs = dict(
        is_multimer=is_multimer,
        enable_ptm=enable_ptm,
        enable_templates=enable_templates,
        inference_chunk_size=4,
        inference_block_size=None,
    )
    feat_cfg_kwargs = dict(
        is_multimer=is_multimer,
    )
    import_kwargs = dict(
        is_multimer=is_multimer,
        enable_ptm=enable_ptm,
        enable_templates=enable_templates,
        fuse_projection_weights=fuse_projection_weights,
    )

    model_name = {
        "deepfold_model_1": "model_1",
        "deepfold_model_2": "model_2",
        "deepfold_model_3": "model_3",
        "deepfold_model_4": "model_4",
        "deepfold_model_5": "model_5",
        "params_model_1": "model_1",
        "params_model_2": "model_2",
        "params_model_3": "model_3",
        "params_model_4": "model_4",
        "params_model_5": "model_5",
        "params_model_1_ptm": "model_1_ptm",
        "params_model_2_ptm": "model_2_ptm",
        "params_model_3_ptm": "model_3_ptm",
        "params_model_4_ptm": "model_4_ptm",
        "params_model_5_ptm": "model_5_ptm",
        "params_model_1_multimer": "model_1_v1",
        "params_model_2_multimer": "model_2_v1",
        "params_model_3_multimer": "model_3_v1",
        "params_model_4_multimer": "model_4_v1",
        "params_model_5_multimer": "model_5_v1",
        "params_model_1_multimer_v2": "model_1_v2",
        "params_model_2_multimer_v2": "model_2_v2",
        "params_model_3_multimer_v2": "model_3_v2",
        "params_model_4_multimer_v2": "model_4_v2",
        "params_model_5_multimer_v2": "model_5_v2",
        "params_model_1_multimer_v3": "model_1_v3",
        "params_model_2_multimer_v3": "model_2_v3",
        "params_model_3_multimer_v3": "model_3_v3",
        "params_model_4_multimer_v3": "model_4_v3",
        "params_model_5_multimer_v3": "model_5_v3",
    }[preset]

    return model_name, (model_cfg_kwargs, feat_cfg_kwargs, import_kwargs)


def _recycle_hook(
    recycle_iter: int,
    feats: dict,
    outputs: dict,
    *,
    output_dirpath: Path | None = None,
    save_recycle: bool = False,
) -> None:
    # Calculate mean plDDT
    outputs["mean_plddt"] = torch.mean(outputs["plddt"])

    print_line = ""
    for k, v in [
        ("mean_plddt", "plDDT"),
        ("ptm_score", "pTM"),
        ("iptm_score", "ipTM"),
        ("weighted_ptm_score", "Confidence"),
    ]:
        if k in outputs:
            print_line += f" {v}={outputs[k]:.3g}"

    logger.info(f"Pred: recycle={recycle_iter}{print_line}")

    if save_recycle:
        batch_np = {
            "residue_index": feats["residue_index"].squeeze(0).cpu().numpy(),
            "aatype": feats["aatype"].squeeze(0).cpu().numpy(),
        }
        if "asym_id" in feats:
            batch_np["asym_id"] = feats["asym_id"].squeeze(0).cpu().numpy()

        outputs_np = {
            "final_atom_mask": outputs["final_atom_mask"].cpu().squeeze(0).numpy(),
            "final_atom_positions": outputs["final_atom_positions"].cpu().squeeze(0).numpy(),
        }

        pdb_filepath = output_dirpath / f"frame_{recycle_iter}.pdb"
        prot = protein.from_prediction(
            processed_features=batch_np,
            result=outputs_np,
            b_factors=outputs["plddt"].cpu().squeeze(0).numpy(),
            remark=f"RECYCLE {recycle_iter} {print_line}",
        )
        with open(pdb_filepath, "w") as fp:
            fp.write(protein.to_pdb(prot))


def predict(args: argparse.Namespace) -> None:
    if (args.seed == -1) and (args.mp_size == 0):
        seed = random.randint(0, NUMPY_SEED_MODULUS)
    else:
        seed = args.seed

    if args.mp_size > 0:
        # Assuming model parallelized prediction:
        dist.initialize()
        process_name = f"dist_process_rank{dist.rank()}"
        device = torch.device(f"cuda:{dist.local_rank()}")
        assert len(dist.train_ranks()) % args.mp_size == 0
        mp.initialize(dap_size=args.mp_size)
        if dist.is_master_process():
            logger.info(f"Initialize distributed prediction: WORLD_SIZE={dist.world_size()} MP_SIZE={mp.size()}")
    else:
        logger.info("Single GPU prediction")
        process_name = "single_process"
        device = torch.device("cuda:0")
        assert args.mp_size == 0

    # Print args:
    if dist.is_master_process():
        logger.info("Arguments:")
        for k, v in vars(args).items():
            logger.info(f"{k}={v}")

    # Set device:
    torch.cuda.set_device(device=device)

    # Distributed warm-up:
    if args.mp_size > 0:
        if mp.is_initialized():
            torch.distributed.barrier(
                group=mp.group(),
                device_ids=[dist.local_rank()],
            )

    # Create output directory:
    args.output_dirpath.mkdir(parents=True, exist_ok=args.force)
    # figures_dirpath = args.output_dirpath / "figures"
    # figures_dirpath.mkdir(parents=True, exist_ok=args.force)

    # Setup suffix:
    suffix = f"_{args.suffix}" if args.suffix else ""

    # Get configs:
    model_name, (model_cfg_kwargs, feat_cfg_kwargs, import_kwargs) = get_preset_opts(args.preset)
    model_config = AlphaFoldConfig.from_preset(**model_cfg_kwargs)
    feat_config = FeaturePipelineConfig.from_preset(
        preset="predict",
        ensemble_seed=seed,
        **feat_cfg_kwargs,
    )

    # Print configs:
    if dist.is_master_process():
        logger.info("Model Config:")
        for k, v in flatten_dict(model_config.to_dict()).items():
            logger.info(f"{k}={v}")
        for k, v in flatten_dict(feat_config.to_dict()).items():
            logger.info(f"{k}={v}")

    # Load input features:
    feats = load_pickle(args.input_features_filepath)
    seqlen = feats["residue_index"].shape[-1]

    # Process input features:
    start_time = time.perf_counter()
    batch = feature_pipeline.FeaturePipeline(config=feat_config).process_features(feats)

    # Template multi-chain mask
    if args.multimer_templates != "":
        templ_idx = list(map(int, args.multimer_templates.split(",")))
        if dist.is_master_process():
            logger.info(f"Multimer template enabled for {','.join(templ_idx)}")
        # Initialize:
        mask = torch.zeros(
            [
                seqlen,
                seqlen,
                feat_config.max_templates,
                feat_config.max_recycling_iters,
            ]
        )
        asym_id = batch["asym_id"][..., 0]
        block_diag_mask = asym_id[..., :, None] == asym_id[..., None, :]

        for i in range(feat_config.max_templates):
            if i in templ_idx:
                mask[..., i, :] = 1.0
            else:
                mask[..., i, :] = block_diag_mask[..., None]

        batch["template_multichain_mask_2d"] = mask

    pipeline_duration = time.perf_counter() - start_time
    if dist.is_master_process():
        logger.info(f"Feature processing done in {pipeline_duration:0.2f} sec")
    batch_last = {k: np.array(v[..., -1].squeeze(0).cpu()) for k, v in batch.items()}

    # Add batch dimension and copy processed features:
    batch = {k: torch.as_tensor(v[None, ...]).to(device=device) for k, v in batch.items()}

    # Disable inductor kernels:
    inductor.disable()

    # Create module:
    model = create_alphafold_module(
        alphafold_config=model_config,
        device=device,
        seed=seed,
    )
    model.eval()
    npz_path = args.params_dirpath / f"{args.preset}.npz"
    import_jax_weights_(
        model=model,
        npz_path=npz_path,
        **import_kwargs,
    )

    if dist.is_master_process():
        logger.info(f"seqlen={seqlen}")
        logger.info("Start inference procedure:")
        tiem_begin = time.perf_counter()
        if args.save_recycle:
            recycle_outpath = args.output_dirpath / f"recycle_{model_name}{suffix}"
            recycle_outpath.mkdir(parents=True, exist_ok=True)
        else:
            recycle_outpath = None
        recycle_hook = partial(
            _recycle_hook,
            output_dirpath=recycle_outpath,
            save_recycle=args.save_recycle,
        )
    else:
        recycle_hook = None
    torch.cuda.reset_peak_memory_stats()

    # Run model:
    out = model(
        batch,
        recycle_hook=recycle_hook,
    )

    if args.mp_size > 0:
        torch.distributed.barrier()

    if dist.is_master_process():
        time_elapsed = time.perf_counter() - tiem_begin
        logger.info(f"Time elapsed: {time_elapsed:.2f} sec")
        logger.info(f"CUDA max memory allocated: {torch.cuda.max_memory_allocated() / 1024/ 1024:.2f} MB")

        # Remove batch dimension:
        # batch_last = tensor_tree_map(
        #     fn=lambda x: np.array(x[..., -1].squeeze(0).cpu()),
        #     tree=batch,
        # )

        # Remove batch dimension:
        out = tensor_tree_map(
            fn=lambda x: np.array(x.squeeze(0).cpu()),
            tree=out,
        )
        logger.info("Save outputs...")
        dump_pickle(out, args.output_dirpath / f"result_{model_name}{suffix}.pkz")

        prot = protein.from_prediction(
            processed_features=batch_last,
            result=out,
            b_factors=out["plddt"],
            remark=f"{args.preset} with seed={seed}",
        )
        with open(args.output_dirpath / f"unrelaxed_{model_name}{suffix}.pdb", "w") as fp:
            fp.write(protein.to_pdb(prot))

    # Exit:
    del model
    if args.mp_size > 0:
        torch.distributed.barrier()
        exit(0)
        if mp.size() >= 2:
            os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


if __name__ == "__main__":
    try:
        predict(parse_args())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt... exit(1)")
        exit(1)
