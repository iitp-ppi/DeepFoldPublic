# Copyright 2024 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from deepfold.config import FeaturePipelineConfig
from deepfold.data import input_pipeline, input_pipeline_multimer

FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


def np_to_tensor_dict(
    np_example: Mapping[str, np.ndarray],
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
    # torch generates warnings if feature is already a torch Tensor
    to_tensor = lambda t: torch.tensor(t) if type(t) != torch.Tensor else t.clone().detach()
    tensor_dict = {k: to_tensor(v) for k, v in np_example.items() if k in features}

    return tensor_dict


def np_example_to_features(
    np_example: FeatureDict,
    config: FeaturePipelineConfig,
):
    np_example = dict(np_example)
    cfg = copy.deepcopy(config)

    seq_length = np_example["seq_length"]
    num_res = int(seq_length[0]) if seq_length.ndim != 0 else int(seq_length)
    if not cfg.residue_cropping_enabled:
        cfg.crop_size = num_res
    feature_names = cfg.feature_names()

    if "deletion_matrix_int" in np_example:
        np_example["deletion_matrix"] = np_example.pop("deletion_matrix_int").astype(np.float32)

    tensor_dict = np_to_tensor_dict(np_example=np_example, features=feature_names)

    with torch.no_grad():
        if cfg.is_multimer:
            features = input_pipeline_multimer.process_tensors_from_config(tensor_dict, cfg)
        else:
            features = input_pipeline.process_tensors_from_config(tensor_dict, cfg)

    if cfg.preset == "train":
        p = torch.rand(1).item()
        use_clamped_fape_value = float(p < cfg.clamp_fape_prob)
        features["use_clamped_fape"] = torch.full(
            size=[cfg.max_recycling_iters + 1],
            fill_value=use_clamped_fape_value,
            dtype=torch.float32,
        )
    else:
        features["use_clamped_fape"] = torch.full(
            size=[cfg.max_recycling_iters + 1],
            fill_value=0.0,
            dtype=torch.float32,
        )

    return {k: v for k, v in features.items()}


class FeaturePipeline:
    def __init__(self, config: FeaturePipelineConfig):
        self.config = config

    def process_features(self, raw_features: FeatureDict) -> FeatureDict:
        return np_example_to_features(np_example=raw_features, config=self.config)
