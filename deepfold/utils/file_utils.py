import functools
import gzip
import os
import pickle
import warnings
from glob import glob
from pathlib import Path
from typing import Any, List, Sequence

__all__ = ["read_text", "load_pickle", "dump_pickle", "find_paths"]


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
            return _load_gz_pkl(path)
        elif ext == ".pkz":
            return _load_pkz(path)
        else:
            return _load_pkl(path)
    except FileNotFoundError:
        warnings.warn(f"File not found: {path}")


def _load_gz_pkl(path: os.PathLike) -> Any:
    with gzip.open(path, "rb") as fp:
        return pickle.load(fp)


def _load_pkz(path: os.PathLike) -> Any:
    return _load_gz_pkl(path)


def _load_pkl(path: os.PathLike) -> Any:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def dump_pickle(obj: Any, path: os.PathLike, level: int = 6) -> None:
    f, ext = os.path.splitext(path)
    assert ext in (".pkl", ".pkz", ".gz")
    if ext == ".pkl":
        path = f + ".pkz"
        warnings.warn(f"Write on '{path}'")
    with gzip.open(path, "wb", compresslevel=level) as fp:
        pickle.dump(obj, fp)


def find_paths(paths: Sequence[os.PathLike]) -> List[Path]:
    found_paths = set()
    for regex in paths:
        path_found = glob(str(regex))
        for p in path_found:
            found_paths.add(str(p))
    found_paths = list(found_paths)
    found_paths.sort()
    return [Path(p) for p in found_paths]
