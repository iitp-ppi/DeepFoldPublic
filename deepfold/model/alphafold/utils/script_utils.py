import logging
import os
from pathlib import Path
from typing import Union, Dict

import torch
import numpy as np
from omegaconf import DictConfig

from deepfold.common import protein
from deepfold.common import residue_constants as rc
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

    return model


# TODO: load_ckpt


def prep_output(
    out: Dict[str, np.ndarray],
    batch: Dict[str, np.ndarray],
    feature_dict: Dict[str, np.ndarray],
    config: DictConfig,
    multimer_ri_gap: int,
) -> protein.Protein:
    plddt = out["plddt"]

    plddt_b_factors = np.repeat(plddt[..., None], rc.atom_type_num, axis=-1)

    # Prepare protein meta-data
    template_domain_names = []
    template_chain_index = None

    if config.globals.use_template and "template_domain_names" in feature_dict:
        template_domain_names = [t.decode() for t in feature_dict["template_domain_names"]]

        # Templates are not shuffled during inferece
        template_domain_names = template_domain_names[: config.data.predict.max_templates]

        if "template_chain_index" in feature_dict:
            template_chain_index = feature_dict["template_chain_index"]
            template_chain_index = template_chain_index[: config.data.predict.max_templates]

    num_recycle = config.globals.max_recycling_iters
    remark = ", ".join(
        [
            f"num_recycle={num_recycle}",
            f"max_templates={config.data.predict.max_templates}",
            f"version={config.info.version}",
        ]
    )

    ri = feature_dict["residue_index"]
    chain_index = (ri - np.arange(ri.shape[0])) / multimer_ri_gap
    chain_index = chain_index.astype(np.int64)
    cur_chain = 0
    prev_chain_max = 0
    for i, c in enumerate(chain_index):
        if c != cur_chain:
            cur_chain = c
            prev_chain_max = i + cur_chain * multimer_ri_gap

        batch["residue_index"][i] -= prev_chain_max

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors,
        chain_index=chain_index,
        remark=remark,
        parents=template_domain_names,
        parents_chain_index=template_chain_index,
    )

    return unrelaxed_protein
