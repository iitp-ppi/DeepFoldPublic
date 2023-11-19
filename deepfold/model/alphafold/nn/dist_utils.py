# Copyright 2023 DeepFold Team

from dataclasses import dataclass
from functools import partial
from typing import Dict, List

import torch.nn as nn
from omegaconf import DictConfig

from deepfold.distributed.legacy import gather, get_pad_size, get_world_size, pad_tensor, scatter
from deepfold.model.alphafold.pipeline.types import TensorDict


@dataclass(frozen=True)
class Shard:
    dim: int
    is_shard: bool = True


# Must be negative int
RECYCLE_DIM = -1
assert RECYCLE_DIM < 0

SHARDING_STRATEGIES: Dict[str, List[int]] = {
    "aatype": [-1],
    "residue_index": [-1],
    "seq_length": None,
    "template_aatype": [-1],
    "template_all_atom_positions": [-3],
    "template_sum_probs": None,
    "template_all_atom_mask": [-2],
    "seq_mask": [-1],
    "msa_mask": [-2, -1],
    "msa_row_mask": [-1],
    "template_mask": None,
    "template_pseudo_beta": [-2],
    "template_pseudo_beta_mask": [-1],
    "template_torsion_angles_sin_cos": [-3],
    "template_alt_torsion_angles_sin_cos": [-3],
    "template_torsion_angles_mask": [-2],
    "atom14_atom_exists": [-2],
    "residx_atom14_to_atom37": [-2],
    "residx_atom37_to_atom14": [-2],
    "atom37_atom_exists": [-2],
    "extra_msa": [-2, -1],
    "extra_msa_mask": [-1],
    "extra_msa_row_mask": [-1],
    "bert_mask": [-2, -1],
    "true_msa": [-2, -1],
    "extra_has_deletion": [-2, -1],
    "extra_deletion_value": [-2, -1],
    "msa_feat": [-3, -2],
    "target_feat": [-2],
    "use_clamped_fape": None,
}


def _get_shard_dim(dim: int):
    assert SHARDING_STRATEGIES < 0

    if RECYCLE_DIM < dim:
        return dim
    else:
        return dim + RECYCLE_DIM


class ScatterFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: TensorDict) -> TensorDict:  # , Dict[str, Padding]]:
        WORLD_SIZE = get_world_size()  # Model parallel group size

        for key, dim in SHARDING_STRATEGIES.items():
            if dim is None:
                continue

            x = batch[key]

            for i, d in enumerate(dim):
                pad_size = get_pad_size(x, d, WORLD_SIZE)
                x = pad_tensor(x, d, pad_size)
                is_shard = len(dim) == 1 or i == 1

                # Don't shard mask
                # Shard to (*, N', N) and (*, S, N')
                if (not key.endswith("_mask")) and is_shard:
                    x = scatter(x, dim=d)

            batch[key] = x

        return batch


class GatherOutputs(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.config = config

    def _collect_and_pad(self, key: str, pad: Shard, batch: TensorDict):
        if key not in batch:
            return

        x = batch[key]
        if pad.is_shard:
            x = gather(x, pad.dim)
        batch[key] = x

    def forward(self, batch: TensorDict) -> TensorDict:
        fn = partial(self._collect_and_pad, batch=batch)

        # Evoformer
        fn("msa", Shard(-2))
        fn("pair", Shard(-3, False))
        fn("pair", Shard(-2))
        fn("single", Shard(-2))

        # Coordinates
        fn("final_atom_positions", Shard(-3))
        fn("final_atom_mask", Shard(-2))
        fn("final_affine_tensor", Shard(-2))

        # Auxilary heads
        fn("lddt_logits", Shard(-2))
        fn("plddt", Shard(-1))
        fn("distogram_logits", Shard(-3))
        fn("masked_msa_logits", Shard(-2))
        fn("experimentally_resolved_logits", Shard(-2))
        if self.config.heads.tm.enabled:
            fn("tm_logits", Shard(-2))
            fn("predicted_tm_score", Shard(-1))

        # Structure module
        fn = partial(self._collect_and_pad, batch=batch["sm"])

        fn("frames", Shard(-2))
        fn("sidechain_frames", Shard(-4))
        fn("unnormalized_angles", Shard(-3))
        fn("angles", Shard(-3))
        fn("positions", Shard(-3))
        fn("states", Shard(-2))
        fn("single", Shard(-2))

        return batch
