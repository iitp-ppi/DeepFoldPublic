import functools
import gzip
import logging
import os
import pickle
import warnings
from typing import Any

__all__ = ["read_text", "load_pickle", "dump_pickle"]


def read_text(path: os.PathLike) -> str:
    return _read_text(str(path))


@functools.lru_cache(16, typed=False)
def _read_text(path: str) -> str:
    try:
        if os.path.splitext(path)[1] == ".gz":
            with gzip.open(path, "rb") as fp:
                return fp.read().decode()
        else:
            with gzip.open(f"{path}.gz", "rb") as fp:
                return fp.read().decode()
    except FileNotFoundError:
        with open(path, "r") as fp:
            return fp.read()


def load_pickle(path: os.PathLike) -> Any:
    try:
        ext = os.path.splitext(path)[1]
        if ext == ".gz":
            _load_gz_pkl(path)
        elif ext == ".pkz":
            _load_pkz(path)
        else:
            _load_pkl(path)
    except FileNotFoundError:
        warnings.warn(f"File not found: {path}")


def _load_gz_pkl(path: os.PathLike) -> Any:
    with gzip.open(path, "rb") as fp:
        return pickle.load(fp)


def _load_pkz(path: os.PathLike) -> Any:
    return _load_gz_pkl(path)


def _load_pkl(path: os.PathLike) -> Any:
    with open(path, "rb") as fp:
        return pickle.load(path)


def dump_pickle(obj: Any, path: os.PathLike, level: int = 6) -> None:
    with gzip.open(f"{path}.pkz", "wb", compresslevel=level) as fp:
        pickle.dump(obj, fp)
