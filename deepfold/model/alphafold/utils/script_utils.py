import logging
import os
from pathlib import Path
from typing import Union

import torch
from omegaconf import DictConfig

from deepfold.model.alphafold.model import AlphaFold
from deepfold.model.alphafold.utils.import_weights import import_jax_weights_

logger = logging.getLogger(__name__)


def load_alphafold(
    config: DictConfig,
    params_dir: Union[str, bytes, os.PathLike],
    device: str = "cuda",
) -> torch.nn.Module:
    npz_path = Path(params_dir) / config.info.params_name
    model = AlphaFold(config=config)
    model = model.eval()
    import_jax_weights_(model, npz_path, version=config.info.version)
    model = model.to(device=device)
    # logger.info(f"Successfully loaded JAX parameters at {npz_path}")

    return model


# TODO: load_ckpt
