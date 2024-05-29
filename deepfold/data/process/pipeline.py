import copy
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from deepfold.config import FeaturePipelineConfig
from deepfold.data.process import monomer, multimer

TensorDict = Dict[str, torch.Tensor]


def example_to_tensor_dict(
    example: dict,
    features: Sequence[str],
) -> TensorDict:
    """Creates dict of tensors from a dict of NumPy arrays.

    Args:
        np_example: A dict of NumPy feature arrays.
        features: A list of strings of feature names to be returned in the dataset.

    Returns:
        A dictionary of features mapping feature names to features. Only the given
        features are returned, all other ones are filtered out.
    """
    # PyTorch generates warnings if feature is already a torch Tensor
    to_tensor = lambda t: torch.tensor(t) if type(t) != torch.Tensor else t.clone().detach()
    tensor_dict = {k: to_tensor(v) for k, v in example.items() if k in features}

    return tensor_dict


def example_to_features(
    example: dict,
    cfg: FeaturePipelineConfig,
) -> TensorDict:
    cfg = copy.deepcopy(cfg)

    seq_len = example["seq_length"]
    num_res = int(seq_len[0]) if seq_len.ndim != 0 else int(seq_len)
    if not cfg.residue_cropping_enabled:
        cfg.crop_size = num_res
    feature_names = cfg.feature_names()

    if "deletion_matrix_int" in example:
        example["deletion_matrix"] = example.pop("deletion_matrix_int").astype(np.float32)

    tensor_dict = example_to_tensor_dict(example, feature_names)

    with torch.no_grad():
        if cfg.is_multimer:
            features = multimer.process_raw_feature_tensors(tensor_dict, cfg)
        else:
            features = monomer.process_raw_feature_tensors(tensor_dict, cfg)

    if cfg.clamped_fape_enabled:
        p = torch.rand(1).item()
        use_clamped_fape = float(p < cfg.clamped_fape_probability)
        features["use_clamped_fape"] = torch.full(
            size=[cfg.max_recycling_iters + 1],
            fill_value=use_clamped_fape,
            dtype=torch.float32,
        )
    else:
        features["use_clamped_fape"] = torch.full(
            size=[cfg.max_recycling_iters + 1],
            fill_value=0.0,
            dtype=torch.float32,
        )

    return {k: v for k, v in features.items()}
