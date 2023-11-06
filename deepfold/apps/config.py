import os
from typing import Union
from pathlib import Path

from omegaconf import OmegaConf, DictConfig


def load(path: Union[str, bytes, os.PathLike]):
    """Load yaml-format configuration."""
    cfg = OmegaConf.load(path)

    return _load_yaml(cfg, path)


def _load_yaml(cfg: DictConfig, path: Union[str, bytes, os.PathLike]) -> DictConfig:
    if "include" in cfg:
        # Load includes
        paths = [Path(path.parent / p) for p in cfg.include]
        del cfg.include
        cfgs = [_load_yaml(OmegaConf.load(p), p) for p in paths]
        # Merge included configs
        cfg = OmegaConf.merge(cfg, *cfgs)

    return cfg
