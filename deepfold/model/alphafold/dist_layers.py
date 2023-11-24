# Copyright 2023 DeepFold Team

from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from deepfold.distributed.legacy import gather, get_pad_size, get_world_size, pad_tensor, scatter
from deepfold.model.alphafold.data.types import TensorDict


@dataclass(frozen=True)
class Shard:
    dim: int
    is_shard: bool = True


# Must be negative int
RECYCLE_DIM = -1
assert RECYCLE_DIM < 0

# 0th entry must be sharded dimension!
NUMRES_SHARDING: Dict[str, List[Optional[int]]] = {
    "aatype": [-1],
    "target_feat": [-2],
    "residue_index": [-1],
    "seq_length": None,
    "seq_mask": [-1],
    "msa_feat": [-2],
    "msa_mask": [-1],
    "msa_row_mask": None,
    "template_aatype": [-1],
    "template_mask": None,
    "template_all_atom_positions": [-3],
    "template_sum_probs": None,
    "template_all_atom_mask": [-2],
    "template_pseudo_beta": [-2],
    "template_pseudo_beta_mask": [-1],
    "template_torsion_angles_sin_cos": [-3],
    "template_alt_torsion_angles_sin_cos": [-3],
    "template_torsion_angles_mask": [-2],
    "atom14_atom_exists": [-2],
    "residx_atom14_to_atom37": [-2],
    "residx_atom37_to_atom14": [-2],
    "atom37_atom_exists": [-2],
    "extra_msa": [-1, -2],
    "extra_msa_mask": [-1, -2],
    "extra_msa_row_mask": [None, -1],
    "extra_has_deletion": [-1, -2],
    "extra_deletion_value": [-1, -2],
    "bert_mask": [-1, -2],
    "true_msa": [-1, -2],
    "use_clamped_fape": None,
}

MSA_SHARDING: Dict[str, Optional[int]] = {
    "msa_feat": -3,
    "msa_mask": -2,
    "msa_row_mask": -1,
    "bert_mask": -2,
    "true_msa": -2,
}


class EvoformerScatter(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, m: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        WORLD_SIZE = get_world_size()

        d = MSA_SHARDING["msa_feat"]
        pad_size = get_pad_size(m, d, WORLD_SIZE)
        m = pad_tensor(m, d, pad_size)

        d = MSA_SHARDING["msa_mask"]
        pad_size = get_pad_size(mask, d, WORLD_SIZE)
        mask = pad_tensor(mask, d, pad_size)

        return m, mask


class ScatterFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: TensorDict) -> TensorDict:
        WORLD_SIZE = get_world_size()  # Model parallel group size

        for key, dim in NUMRES_SHARDING.items():
            if (dim is None) or (key not in batch):
                continue

            x = batch[key]

            for i, d in enumerate(dim):
                if d is None:
                    continue
                pad_size = get_pad_size(x, d, WORLD_SIZE)
                x = pad_tensor(x, d, pad_size)
                is_shard = len(dim) == 1 or i == 0  # 0th

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
            fn("tm_logits", Shard(-3))
            fn("aligned_confidence_probs", Shard(-3))
            fn("predicted_aligned_error", Shard(-2))
            # Skip "max_predicted_aligned_error"
            # Skip "predicted_tm_score"

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
