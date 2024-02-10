# Copyright 2024 DeepFold Team


"""Process examples to input features."""


from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

FeatureDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


def np_to_tensor_dict(
    np_example: FeatureDict,
    keys: Sequence[str],
) -> TensorDict:
    r"""
    Creates dict of tensors from a dict of ndarrays.

    Args:
        np_example:
            A dict of np.ndarray feature arrays.
        keys:
            A list of strings of feature names to be filtered.

    Returns:
        A dictionary of features.
    """

    to_tensor = lambda t: torch.tensor(t) if type(t) != torch.Tensor else t.clone()
    tensor_dict = {k: to_tensor(v) for k, v in np_example.items() for k in keys}

    return tensor_dict


def make_data_config(
    config: DictConfig,
    mode: str,
) -> Tuple[DictConfig, List[str]]:
    r"""Returns configurations and feature names."""

    cfg = config.copy()
    mode_cfg = cfg[mode]

    feature_names = []

    return cfg, feature_names


def np_example_to_features(
    np_example: FeatureDict,
    config: DictConfig,
    mode: str,
    is_multimer: bool = False,
) -> FeatureDict:
    r"""Convert an example to input features."""

    np_example = dict(np_example)
    cfg, feature_names = make_data_config(config, mode)

    return {}


class FeaturePipeline:
    def __init__(self, config: DictConfig):
        self.config = config.copy()

    def process_features(
        self,
        np_example: FeatureDict,
        mode: str,
        is_multimer: bool = False,
    ) -> FeatureDict:
        return np_example_to_features(
            np_example,
            self.config,
            mode,
            is_multimer=is_multimer,
        )
