# Copyright 2023 DeepFold Team

import argparse
import logging
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

import deepfold.distributed.legacy as dist
from deepfold.apps.config_utils import load as load_config
from deepfold.common import protein
from deepfold.model.alphafold.data.pipeline_monomer import FeaturePipeline
from deepfold.model.alphafold.data.types import FeatureDict
from deepfold.model.alphafold.utils.script_utils import load_alphafold, prep_output
from deepfold.utils.tensor_utils import tensor_tree_map

DEBUG_MODE = "DEBUG" in os.environ and os.environ["DEBUG"] != "0"

logger_cfg = {
    "format": r"%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    "datefmt": r"%Y-%m-%d %H:%M:%S",
}

if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG, **logger_cfg)
else:
    logging.basicConfig(level=logging.INFO, **logger_cfg)

logger = logging.getLogger(__name__)


torch.set_grad_enabled(False)


def run_model(local_rank: int, kwargs: Dict[str, Any]):
    world_size = kwargs["world_size"]
    random_seed = kwargs["random_seed"]

    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = kwargs["MASTER_ADDR"]
    os.environ["MASTER_PORT"] = os.environ["MASTER_PORT"]

    dist.init_distributed(
        tensor_model_parallel_size=world_size,
        random_seed=random_seed,
    )
    assert dist.is_initialized()
    assert dist.is_nccl_available()

    model = load_alphafold(config=kwargs["config"], params_dir=kwargs["params_dir"], device="cuda")
    logger.info(f"[{local_rank}] Model loaded")

    processed_feature_dict = {k: torch.as_tensor(v, device="cuda") for k, v in kwargs["processed_feature_dict"].items()}
    logger.info(f"[{local_rank}] Input feature loaded")

    if dist.is_initialized():
        logger.debug(f"[{local_rank}] System initialized")
    else:
        raise RuntimeError(f"Local rank {local_rank} not initialized")

    dist.barrier()
    if dist.is_master():
        logger.info("Start inference")

    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()

    out = model(processed_feature_dict)

    t = time.perf_counter() - t
    torch.cuda.synchronize()
    dist.barrier()

    if dist.is_master():
        out = tensor_tree_map(lambda x: x.detach().cpu().numpy(), out)

        logger.info(f"Inference finished: {t} sec")
        logger.info(f"Max GPU memory: {torch.cuda.max_memory_allocated() * 1e-9:0.3f} GB")

        logger.info("Finish inference")

        queue: mp.Queue = kwargs["queue"]
        queue.put(out)


def predict_structure(
    config: DictConfig,
    feature_dict: FeatureDict,
    output_dir_base: Path,
    params_dir: Path,
    world_size: int,
    name: Optional[str] = None,
    random_seed: int = 0,
) -> Dict[str, np.ndarray]:
    if name is None:
        try:
            name = feature_dict["domain_name"][0].decode().strip()
        except Exception as e:
            logger.debug(str(e))
            name = "protein"

    logger.info(f"Predict target '{name}'")
    logger.info(f"Use model version '{config.info.version}")
    logger.info(f"Random seed {random_seed}")

    # Output name
    try:
        output_name = f"{config.info.version}_n{world_size}"
    except:
        output_name = f"target_n{world_size}"

    # Randomness
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    deterministic = True
    if deterministic:
        logger.info("Deterministic mode enabled")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Output directory
    output_dir = output_dir_base
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    logger.debug("Raw features:")
    for k, v in feature_dict.items():
        logger.debug(f"{k}: {v.dtype}{list(v.shape)}")
    logger.debug("")

    # Process features
    if "template_all_atom_masks" in feature_dict:
        logger.debug("Key 'template_all_atom_masks' found in input features. Rename to 'template_all_atom_mask'")
        feature_dict["template_all_atom_mask"] = feature_dict["template_all_atom_masks"]
        del feature_dict["template_all_atom_masks"]

    processed_feature_dict = FeaturePipeline(config.data).process(feature_dict, mode="predict")
    processed_feature_dict = {k: torch.as_tensor(v, device="cpu") for k, v in processed_feature_dict.items()}

    # Save processed features
    processed_feature_dict_np = tensor_tree_map(lambda x: x.cpu().numpy(), processed_feature_dict)
    processed_feature_path = output_dir / f"{output_name}_input_dict.pkl"
    logger.info(f"Write processed features to '{processed_feature_path}'")
    with open(processed_feature_path, "wb") as fp:
        pickle.dump(processed_feature_dict_np, fp)

    logger.debug("Processesd features:")
    for k, v in processed_feature_dict.items():
        logger.debug(f"{k}: {v.dtype}{list(v.shape)}")
    logger.debug("")

    # Prepare multi-process system
    manager = mp.Manager()
    queue = manager.Queue()

    logger.info(f"Spawn {world_size} procs")
    mp.spawn(
        run_model,
        args=[
            {
                "queue": queue,
                "config": config,
                "output_dir": output_dir,
                "params_dir": params_dir,
                "processed_feature_dict": processed_feature_dict,
                "random_seed": random_seed,
                "world_size": world_size,
                "MASTER_ADDR": "localhost",
            }
        ],
        nprocs=world_size,
    )

    # Get returned batch (from the master rank)
    batch = queue.get()

    # Take last one (Toss out the recycling dimensions)
    processed_feature_dict = tensor_tree_map(lambda x: x[..., -1].cpu().numpy(), processed_feature_dict)

    # Save outputs
    unrelaxed_protein = prep_output(batch, processed_feature_dict, feature_dict, config, multimer_ri_gap=200)
    unrelaxed_file_suffix = "_unrelaxed.pdb"
    unrelaxed_output_path = os.path.join(output_dir, f"{output_name}{unrelaxed_file_suffix}")

    with open(unrelaxed_output_path, "w") as fp:
        fp.write(protein.to_pdb(unrelaxed_protein))

    logger.info(f"Output written to {unrelaxed_output_path}")

    output_dict_path = os.path.join(output_dir, f"{output_name}_output_dict.pkl")
    with open(output_dict_path, "wb") as fp:
        pickle.dump(batch, fp, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Model output written to {output_dict_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_pkl",
        "-f",
        type=str,
        dest="features",
        required=True,
        help="Path to pickled features",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to config yaml",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="_output",
        help="Name of the directory to save outputs",
    )
    parser.add_argument(
        "--jax_params_dir",
        "-p",
        type=str,
        default="_data/params",
        help="Path to JAX parameters directory",
    )
    parser.add_argument(
        "--nprocs",
        "-nt",
        type=int,
        default=1,
        help="How many GPUs to use",
    )
    parser.add_argument(
        "--deterministic",
        action="store_false",
        help="Turn on deterministic mode",
    )
    parser.add_argument(
        "--data_random_seed",
        type=int,
        default=None,
        help="Random seed for preprocess input features",
    )
    parser.add_argument(
        "--options",
        type=str,
        action="store",
        dest="options",
        nargs="*",
        default=[""],
        help="Inject options to the config from dot-lists",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    options = OmegaConf.from_dotlist(args.options)

    if not options.is_empty():
        logger.info("Inject options to the configuration")
    for line in OmegaConf.to_yaml(options).splitlines():
        logger.info(line)
    cfg = OmegaConf.merge(cfg, options)

    output_dir_base = Path(args.output_dir)
    jax_params_dir = Path(args.jax_params_dir)

    logger.info(f"Load features from '{args.features}'")
    with open(args.features, "rb") as fp:
        feature_dict = pickle.load(fp)

    if "MASTER_PORT" not in os.environ:
        random.seed()
        random_port = random.randrange(2**14, 2**16)
        os.environ["MASTER_PORT"] = str(random_port)
        logger.info(f"MASTER_PORT is not set. Set to {random_port}")

    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)

    ngpus = args.nprocs

    total_gpus = torch.cuda.device_count()

    if ngpus > total_gpus:
        raise RuntimeError(f"Number of GPUSs required ({ngpus}) is larger than total number of GPUs ({total_gpus})")

    predict_structure(
        config=cfg,
        feature_dict=feature_dict,
        output_dir_base=output_dir_base,
        params_dir=jax_params_dir,
        random_seed=random_seed,
        world_size=args.nprocs,
    )


if __name__ == "__main__":
    main()
