from typing import Dict, Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from tests.alphafold_config import model_config


def get_alphafold_config():
    config = model_config("model_1_ptm")
    config.model.global_config.deterministic = True
    return config


_PARAM_PATH = "_data/params/params_model_1_ptm.npz"


def flat_params_to_haiku(params: Mapping[str, np.ndarray]) -> hk.Params:
    """Convert a dictionary of NumPy arrays to Haiku parameters."""
    hk_params = {}
    for path, array in params.items():
        scope, name = path.split("//")
        if scope not in hk_params:
            hk_params[scope] = {}
        hk_params[scope][name] = jnp.array(array)

    return hk_params


_ORIG_WEIGHTS = None


def _get_orig_weights():
    global _ORIG_WEIGHTS
    if _ORIG_WEIGHTS is None:
        _ORIG_WEIGHTS = np.load(_PARAM_PATH)

    return _ORIG_WEIGHTS


def _remove_key_prefix(d, prefix):
    for k, v in list(d.items()):
        if k.startswith(prefix):
            d.pop(k)
            d[k[len(prefix) :]] = v


def fetch_alphafold_module_weights_to_haiku(weight_path: str) -> hk.Params:
    orig_weights = _get_orig_weights()
    params = {k: v for k, v in orig_weights.items() if weight_path in k}
    if "/" in weight_path:
        spl = weight_path.split("/")
        spl = spl if len(spl[-1]) != 0 else spl[:-1]
        # module_name = spl[-1]
        prefix = "/".join(spl[:-1]) + "/"
        _remove_key_prefix(params, prefix)

    params = flat_params_to_haiku(params)

    return params


def fetch_alphafold_module_weights_to_dict(weight_path: str) -> Dict[str, np.ndarray]:
    orig_weights = _get_orig_weights()
    params = {k: v for k, v in orig_weights.items() if weight_path in k}
    if "/" in weight_path:
        spl = weight_path.split("/")
        spl = spl if len(spl[-1]) != 0 else spl[:-1]
        # module_name = spl[-1]
        prefix = "/".join(spl[:-1])
        _remove_key_prefix(params, prefix)

    return params
