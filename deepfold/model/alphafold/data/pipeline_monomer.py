# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited


import logging

from omegaconf import DictConfig

from deepfold.model.alphafold.data.proc_feats import np_example_to_features
from deepfold.model.alphafold.data.types import FeatureDict, TensorDict

logger = logging.getLogger(__name__)


class FeaturePipeline:
    def __init__(self, cfg: DictConfig) -> None:
        self.config = cfg

    def process(self, raw_features: FeatureDict, mode: str = "predict") -> TensorDict:
        return np_example_to_features(np_example=raw_features, cfg=self.config, mode=mode)
