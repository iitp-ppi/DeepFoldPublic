# Copyright 2024 DeepFold Team


"""DeepFold2 model configuration."""


import logging
from dataclasses import asdict, dataclass
from typing import Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ConfigBase:
    def to_omegaconf(self):
        return OmegaConf.structured(self)


@dataclass(kw_only=True)
class DeepFoldConfig(ConfigBase):
    r"""
    This is the configuration class to store the configuration of a DeepFold model. It is used to instantiate a
    DeepFold model according to the specified arguments, defining model architecture.
    """

    fp16: bool = False
    bf16: bool = False

    chunk_size: int = 4
    max_recycles: int = 3
    eps: float = 1e-8
    inf: float = 3e4

    embedder: "EmbedderConfig" = None
    evoformer: "EvoformerConfig" = None
    structure_module: "StructureModuleConfig" = None
    auxiliary_heads: "AuxiliaryHeadsConfig" = None

    def __post_init__(self):
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 cannot be enabled simultaneously")

        if self.embedder is None:
            self.embedder = EmbedderConfig()
        elif isinstance(self.trunk, (dict, DictConfig)):
            self.embedder = EmbedderConfig(**self.embedder)


@dataclass(kw_only=True)
class EmbedderConfig(ConfigBase):
    pass


@dataclass(kw_only=True)
class InputEmbedderConfig(ConfigBase):
    target_feature_dim: int = 22  # 21
    msa_feature_dim: int = 49
    pair_representation_dim: int = 128
    msa_representation_dim: int = 256
    relative_position_bins: int = 32  # 33


@dataclass(kw_only=True)
class EvoformerConfig(ConfigBase):
    pass


@dataclass(kw_only=True)
class StructureModuleConfig(ConfigBase):
    pass


@dataclass(kw_only=True)
class AuxiliaryHeadsConfig(ConfigBase):
    pass


@dataclass(kw_only=True)
class DeepFoldLossConfig(ConfigBase):
    pass
