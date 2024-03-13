import gzip
import logging
import os
import pickle
from typing import Any

__all__ = ["read_text", "load_pickle", "dump_pickle"]

logger = logging.getLogger(__name__)


def read_text(path: os.PathLike) -> str:
    try:
        with open(path, "r") as fp:
            return fp.read()
    except FileNotFoundError:
        logger.debug(f"File not found: {path}")
        with gzip.open(f"{path}.gz", "rb") as fp:
            return fp.read().decode()


def load_pickle(path: os.PathLike) -> Any:
    try:
        with open(path, "rb") as fp:
            return pickle.load(fp)
    except FileNotFoundError:
        logger.debug(f"File not found: {path}")
        with gzip.open(f"{path}.gz", "rb") as fp:
            return pickle.load(fp)


def dump_pickle(obj: Any, path: os.PathLike) -> None:
    with gzip.open(f"{path}.gz", "rb") as fp:
        pickle.dump(obj, fp)
