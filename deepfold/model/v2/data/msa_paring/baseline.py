# Copyright 2024 DeepFold Team


import collections
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pad
import scipy.linalg as linalg

import deepfold.common.residue_constants as rc
from deepfold.model.v2.data.ops import FeatureDict

MSA_GAP_IDX = rc.restypes_with_x_and_gap.index("-")
SEQUENCE_GAP_CUTOFF = 0.5
SEQUENCE_SIMILARITY_CUTOFF = 0.9


MSA_PAD_VALUES = {
    "msa_all_seq": MSA_GAP_IDX,
    "msa_mask_all_seq": 1,
    "deletion_matrix_all_seq": 0,
    "deletion_matrix_int_all_seq": 0,
    "msa": MSA_GAP_IDX,
    "msa_mask": 1,
    "deletion_matrix": 0,
    "deletion_matrix_int": 0,
}

MSA_FEATURES = (
    "msa",
    "msa_mask",
    "deletion_matrix",
    "deletion_matrix_int",
)
SEQ_FEATURES = (
    "residue_index",
    "aatype",
    "all_atom_positions",
    "all_atom_mask",
    "seq_mask",
    "between_segment_residues",  # Deprecated
    "has_alt_locations",
    "has_hetatoms",
    "asym_id",
    "entity_id",
    "sym_id",
    "entity_mask",
    "deletion_mean",
    "prediction_atom_mask",
    "literature_positions",
    "atom_indices_to_group_indices",
    "rigid_group_default_frame",
)
TEMPLATE_FEATURES = (
    "template_aatype",
    "template_all_atom_positions",
    "template_all_atom_mask",
)
CHAIN_FEATURES = (
    "num_alignments",
    "seq_length",
)


