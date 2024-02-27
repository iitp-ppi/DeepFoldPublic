import contextlib
import os
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from deepfold.common import residue_constants as rc

FeatureDict = Dict[str, np.ndarray]


@contextlib.contextmanager
def temp_fasta_file(fasta_str: str):
    with tempfile.NamedTemporaryFile("w", suffix=".fasta") as fasta_file:
        fasta_file.write(fasta_str)
        fasta_file.seek(0)
        yield fasta_file.name


def convert_monomer_features(monomer_features: FeatureDict) -> FeatureDict:
    """Reshapes and modifies monomer features for multimer models."""

    unnecessary_leading_dim_feats = ["sequence", "domain_name", "num_alignments", "seq_length"]

    converted = {}
    for key, feat in monomer_features.items():
        if key in unnecessary_leading_dim_feats:
            feat = np.asarray(feat[0], dtype=feat.dtype)
        elif key == "aatype":
            feat = np.argmax(feat, axis=-1).astype(np.int32)
        elif key == "template_aatype":
            feat = np.argmax(feat, axis=-1).astype(np.int32)
            new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
            feat = np.take(new_order_list, feat.astype(np.int32), axis=0)
        elif key == "template_all_atom_masks":
            key = "template_all_atom_mask"
        converted[key] = feat

    return converted


def int_id_to_str_id(num: int) -> str:
    """
    Encodes a number as a string, using reverse spreadsheet style naming.

    Args:
      num: A positive integer.

    Returns:
      A string that encodes the positive integer using reverse spreadsheet style,
      naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
      usual way to encode chain IDs in mmCIF files.
    """
    if num <= 0:
        raise ValueError(f"Only positive integers allowed, got {num}.")

    num = num - 1  # 1-based indexing.
    output = []
    while num >= 0:
        output.append(chr(num % 26 + ord("A")))
        num = num // 26 - 1
    return "".join(output)


def add_assembly_features(all_chain_features: Dict[str, FeatureDict]) -> Dict[str, FeatureDict]:
    """
    Add features to distinguish between chains.

    Args:
        all_chain_features:
            A dictionary which maps chain_id to a dictionary of features for each chain.

    Returns:
        all_chain_features:
            A dictionary which maps strings of the form
            `<seq_id>_<sym_id>` to the corresponding chain features. E.g. two
            chains from a homodimer would have keys A_1 and A_2. Two chains from a
            heterodimer would have keys A_1 and B_1.
    """
    # Group the chains by sequence.
    seq_to_entity_id = {}
    grouped_chains = defaultdict(list)
    for chain_id, chain_features in all_chain_features.items():
        seq = str(chain_features["sequence"])
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains[seq_to_entity_id[seq]].append(chain_features)

    new_all_chain_features = {}
    chain_id = 1
    for entity_id, group_chain_features in grouped_chains.items():
        num_sym = len(group_chain_features)  #
        for sym_id, chain_features in enumerate(group_chain_features, start=1):
            new_all_chain_features[f"{int_id_to_str_id(entity_id)}_{sym_id}"] = chain_features
            seq_length = chain_features["seq_length"]
            chain_features["asym_id"] = (chain_id * np.ones(seq_length)).astype(np.int64)
            chain_features["sym_id"] = (sym_id * np.ones(seq_length)).astype(np.int64)
            chain_features["entity_id"] = (entity_id * np.ones(seq_length)).astype(np.int64)
            chain_features["num_sym"] = (num_sym * np.ones(seq_length)).astype(np.int64)  # v2
            chain_id += 1

    return new_all_chain_features


def pad_msa(np_example: FeatureDict, min_num_seq: int) -> FeatureDict:
    np_example = dict(np_example)
    num_seq = np_example["msa"].shape[0]
    if num_seq < min_num_seq:
        for feat in ("msa", "deletion_matrix", "bert_mask", "msa_mask"):
            np_example[feat] = np.pad(np_example[feat], ((0, min_num_seq - num_seq), (0, 0)))
        np_example["cluster_bias_mask"] = np.pad(np_example["cluster_bias_mask"], ((0, min_num_seq - num_seq),))
    return np_example
