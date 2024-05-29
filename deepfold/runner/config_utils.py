import os
from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf


def load(path: Union[str, bytes, os.PathLike]):
    """Load yaml-format configuration."""
    path = Path(str(path))
    cfg = OmegaConf.load(path)

    return _load_yaml(cfg, path)


def _load_yaml(cfg: DictConfig, path: Path) -> DictConfig:
    if "include" in cfg:
        # Load includes
        paths = [Path(path.parent / p) for p in cfg.include]
        del cfg.include
        cfgs = [_load_yaml(OmegaConf.load(p), p) for p in paths]
        # Merge included configs
        cfg = OmegaConf.merge(*cfgs, cfg)

    return cfg
