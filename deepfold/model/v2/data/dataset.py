# Copyright 2024 DeepFold Team


"""Dataloader modules."""


import copy
import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

from deepfold.model.v2.data import utils
from deepfold.model.v2.data.ops import FeatureDict, TensorDict
from deepfold.model.v2.data.process import (
    add_assembly_features,
    convert_monomer_features,
    merge_msas,
    pair_and_merge,
    post_process,
    process_features,
    process_labels,
)

logger = logging.getLogger(__name__)


def make_data_config(
    config: DictConfig,
    mode: str,
    num_res: int,
) -> Tuple[DictConfig, List[str]]:
    """
    Generates a data configuration and a list of feature names based on the given mode and settings.

    This function adapts the provided configuration for a specific mode (e.g., 'train', 'test'). It sets the crop size
    if not already specified, and compiles a list of feature names based on various conditions in the configuration.

    Args:
        config (DictConfig): The original configuration object from OmegaConf.
        mode (str): The mode for which the configuration is being prepared (e.g., 'train', 'test').
        num_res (int): The number of residues to use in case the crop size is not specified in the configuration.

    Returns:
        Tuple[DictConfig, List[str]]: A tuple containing the updated configuration and the list of feature names.
    """

    # Copy the original configuration to avoid modifying it directly
    cfg = config.copy()

    # Access the specific configuration for the given mode
    mode_cfg = cfg[mode]

    # Set the crop size to num_res if it's not already specified
    if mode_cfg.crop_size is None:
        mode_cfg.crop_size = num_res

    # Start with unsupervised and recycling features from the common configuration
    feature_names = list(zip(cfg.common.unsupervised_features, cfg.common.recycling_features))

    # Add template features if they are used
    if cfg.common.use_templates:
        feature_names += list(cfg.common.template_features)

    # Include multimer features for multimer configurations
    if cfg.common.is_multimer:
        feature_names += list(cfg.common.multimer_features)

    # Add supervised features if the mode is supervised
    if cfg[mode].supervised:
        feature_names += list(cfg.supervised.supervised_features)

    return cfg, feature_names
