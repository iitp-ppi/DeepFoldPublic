# Copyright 2023 DeepFold Team
# Copyright 2021 DeepMind Technologies Limited

import logging
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np

from deepfold.common import residue_constants as rc
from deepfold.data import msa_identifiers, parsers

logger = logging.getLogger(__name__)


FeatureDict = MutableMapping[str, np.ndarray]


def make_sequence_feature(
    sequence: str,
    description: str,
    num_res: int,
) -> FeatureDict:
    """
    Construct a feature dict of sequence features.
    """
    features = {}

    features["aatype"] = rc.sequence_to_onehot(
        sequence=sequence,
        mapping=rc.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array([description.encode()], dtype=np.object_)
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)  # Start from zero
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array([sequence.encode()], dtype=np.object_)

    return features


def make_msa_features(msas: Sequence[parsers.MSA]) -> FeatureDict:
    """
    Construct a feature dict of MSA features.
    """
    if not msas:
        raise ValueError("At least one MSA must be provided")

    int_msa = []
    deletion_matrix = []
    species_ids = []
    seen_sequences = set()

    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(f"MSA {msa_index} must contain at least one sequence")
        for seq_index, seq in enumerate(msa.sequences):
            if seq in seen_sequences:
                continue
            seen_sequences.add(seq)
            int_msa.append([rc.HHBLITS_AA_TO_ID[res] for res in seq])
            deletion_matrix.append(msa.deletion_matrix[seq_index])
            identifiers = msa_identifiers.get_identifiers(msa.descriptions[seq_index])
            species_ids.append(identifiers.species_id.encode())

    num_res = len(msas[0].sequences[0])
    num_alignments = len(int_msa)

    features = {}

    features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
    features["msa"] = np.array(int_msa, dtype=np.int32)
    features["num_alignments"] = np.array([num_alignments] * num_res, dtype=np.int32)
    features["msa_species_identifiers"] = np.array(species_ids, dtype=np.object_)

    return features


# TODO: process_templates

# TODO: run_msa_tool

# TODO: AlphaFoldPipeline
