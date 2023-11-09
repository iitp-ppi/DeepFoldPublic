# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited

import logging
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np

from deepfold.common import residue_constants as rc
from deepfold.data import msa_identifiers, parsers

logger = logging.getLogger(__name__)


FeatureDict = MutableMapping[str, np.ndarray]


def empty_template_feats(n_res: int) -> FeatureDict:
    return {
        "template_aatype": np.zeros((0, n_res)).astype(np.int64),
        "template_all_atom_positions": np.zeros((0, n_res, 37, 3)).astype(np.float32),
        "template_sum_probs": np.zeros((0, 1)).astype(np.float32),
        "template_all_atom_mask": np.zeros((0, n_res, 37)).astype(np.float32),
    }


def make_template_features(
    input_sequence: str,
    hits: Sequence[Any],
    template_featurizer: Any,
    query_pdb_code: Optional[str] = None,
    query_release_date: Optional[str] = None,
) -> FeatureDict:
    hits_cat = sum(hits.values(), [])
    if len(hits_cat) == 0 or template_featurizer is None:
        template_features = empty_template_feats(len(input_sequence))
    else:
        templates_result = template_featurizer.get_templates(
            query_sequence=input_sequence,
            query_pdb_code=query_pdb_code,
            query_release_date=query_release_date,
            hits=hits_cat,
        )
        template_features = templates_result.features

        # TODO: The template featurizer doesn't format empty template features properly.
        if template_features["template_aatype"].shape[0] == 0:
            template_features = empty_template_feats(len(input_sequence))

    return template_features


def unify_template_features(template_feature_list: Sequence[FeatureDict]) -> FeatureDict:
    out_dicts = []
    seq_lens = [fd["template_aatype"].shape[1] for fd in template_feature_list]
    for i, fd in enumerate(template_feature_list):
        out_dict = {}
        n_templates, n_res = fd["template_aatype"].shape[:2]
        for k, v in fd.items():
            seq_keys = [
                "template_aatype",
                "template_all_atom_positions",
                "template_all_atom_mask",
            ]
            if k in seq_keys:
                new_shape = list(v.shape)
                assert new_shape[1] == n_res
                new_shape[1] = sum(seq_lens)
                new_array = np.zeros(new_shape, dtype=v.dtype)

                if k == "template_aatype":
                    new_array[..., rc.HHBLITS_AA_TO_ID["-"]] = 1

                offset = sum(seq_lens[:i])
                new_array[:, offset : offset + seq_lens[i]] = v
                out_dict[k] = new_array
            else:
                out_dict[k] = v

        chain_indices = np.array(n_templates * [i])
        out_dict["template_chain_index"] = chain_indices

        if n_templates != 0:
            out_dicts.append(out_dict)

    if len(out_dicts) > 0:
        out_dict = {k: np.concatenate([od[k] for od in out_dicts]) for k in out_dicts[0]}
    else:
        out_dict = empty_template_feats(sum(seq_lens))

    return out_dict


# TODO: make_sequence_features

# TODO: make_mmcif_features

# TODO: make_protein_featuers

# TODO: make_pdb_features

# TODO: make_msa_features

# TODO: make_dummy_msa_features

# TODO: make_sequence_features_with_custom_template

# TODO: AlignmentRunner

# TODO: DataPipeline
