# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 AlQuraishi Laboratory
# Copyright 2023 NVIDIA CORPORATION
# Copyright 2024 DeepFold Team


from typing import List, Sequence

import numpy as np

from deepfold.common import protein
from deepfold.common import residue_constants as rc
from deepfold.data.search.mmcif import zero_center_atom_positions
from deepfold.data.search.parsers import parse_a3m, parse_hhr, parse_hmmsearch_sto
from deepfold.data.search.templates import TemplateHit, TemplateHitFeaturizer
from deepfold.utils.datetime_utils import datetime_from_string


def create_sequence_features(sequence: str, domain_name: str) -> dict:
    seqlen = len(sequence)  # num residues
    sequence_features = {}
    sequence_features["aatype"] = rc.sequence_to_onehot(sequence=sequence, mapping=rc.restype_order_with_x, map_unknown_to_x=True)
    sequence_features["between_segment_residues"] = np.zeros(shape=(seqlen), dtype=np.int32)
    sequence_features["domain_name"] = np.array([domain_name.encode("utf-8")], dtype=np.object_)
    sequence_features["residue_index"] = np.arange(seqlen, dtype=np.int32)
    sequence_features["seq_length"] = np.full(shape=(seqlen), fill_value=seqlen, dtype=np.int32)
    sequence_features["sequence"] = np.array([sequence.encode("utf-8")], dtype=np.object_)
    return sequence_features


def create_mmcif_features(
    mmcif_dict: dict,
    author_chain_id: str,
    zero_center: bool = False,
) -> dict:
    mmcif_features = {}

    pdb_chain_id = mmcif_dict["pdb_id"] + author_chain_id
    sequence = mmcif_dict["sequences"][author_chain_id]

    sequence_features = create_sequence_features(sequence=sequence, domain_name=pdb_chain_id)
    mmcif_features.update(sequence_features)

    all_atom_positions = mmcif_dict["atoms"][author_chain_id]["all_atom_positions"]
    all_atom_mask = mmcif_dict["atoms"][author_chain_id]["all_atom_mask"]
    if zero_center:
        all_atom_positions = zero_center_atom_positions(all_atom_positions=all_atom_positions, all_atom_mask=all_atom_mask)
    mmcif_features["all_atom_positions"] = all_atom_positions.astype(np.float32)
    mmcif_features["all_atom_mask"] = all_atom_mask.astype(np.float32)

    mmcif_features["resolution"] = np.array([mmcif_dict["resolution"]], dtype=np.float32)
    mmcif_features["release_date"] = np.array([mmcif_dict["release_date"].encode("utf-8")], dtype=np.object_)
    mmcif_features["is_distillation"] = np.array(0.0, dtype=np.float32)

    return mmcif_features


def _aatype_to_str_sequence(aatype: Sequence[int]) -> str:
    return "".join([rc.restypes_with_x[aatype[i]] for i in range(len(aatype))])


def create_protein_features(
    protein_object: protein.Protein,
    description: str,
    is_distillation: bool = False,
) -> dict:
    pdb_feats = {}
    aatype = list(protein_object.aatype)  # [NUM_RES]
    sequence = _aatype_to_str_sequence(aatype)
    pdb_feats.update(create_sequence_features(sequence=sequence, domain_name=description))
    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask
    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)
    pdb_feats["resolution"] = np.array([0.0]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(1.0 if is_distillation else 0.0).astype(np.float32)
    return pdb_feats


def create_pdb_features(
    protein_object: protein.Protein,
    description: str,
    is_distillation: bool = True,
    confidence_threshold: float = 50.0,
) -> dict:
    pdb_feats = create_protein_features(protein_object, description, is_distillation=True)
    if is_distillation:
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]
    return pdb_feats


def create_template_features(
    sequence: str,  # Query sequence
    template_hits: Sequence[TemplateHit],
    template_hit_featurizer: TemplateHitFeaturizer,
    max_release_date: str,
    pdb_id: str | None = None,  # Optional query pdb id
    shuffling_seed: int | None = None,
) -> dict:
    query_release_date = datetime_from_string(max_release_date, r"%Y-%m-%d")
    template_features = template_hit_featurizer.get_template_features(
        query_sequence=sequence,
        template_hits=list(template_hits),
        max_template_date=query_release_date,
        query_pdb_id=pdb_id,
        shuffling_seed=shuffling_seed,
    )
    return template_features


def create_template_features_from_hhr_string(
    sequence: str,
    hhr_string: str,
    template_hit_featurizer: TemplateHitFeaturizer,
    release_date: str,
    pdb_id: str | None = None,
    shuffling_seed: int | None = None,
) -> dict:
    template_hits = parse_hhr(hhr_string)
    template_features = create_template_features(
        sequence=sequence,
        template_hits=template_hits,
        template_hit_featurizer=template_hit_featurizer,
        max_release_date=release_date,
        pdb_id=pdb_id,
        shuffling_seed=shuffling_seed,
    )
    return template_features


def create_template_features_from_hmmsearch_sto_string(
    sequence: str,
    sto_string: str,
    template_hit_featurizer: TemplateHitFeaturizer,
    release_date: str,
    pdb_id: str | None = None,
    shuffling_seed: int | None = None,
) -> dict:
    template_hits = parse_hmmsearch_sto(sequence, sto_string)
    template_features = create_template_features(
        sequence=sequence,
        template_hits=template_hits,
        template_hit_featurizer=template_hit_featurizer,
        max_release_date=release_date,
        pdb_id=pdb_id,
        shuffling_seed=shuffling_seed,
    )
    return template_features


def create_msa_features(
    sequence: str,
    a3m_strings: List[str],
    use_identifiers: bool = False,
) -> dict:
    msas = []
    deletion_matrices = []
    descriptions = []
    for a3m_string in a3m_strings:
        if not a3m_string:
            continue
        msa, deletion_matrix, desc = parse_a3m(a3m_string)
        msas.append(msa)
        deletion_matrices.append(deletion_matrix)
        descriptions.extend(desc)

    if len(msas) == 0:
        msas.append([sequence])
        deletion_matrices.append([[0 for _ in sequence]])
        descriptions.append("")

    int_msa = []
    deletion_matrix = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(f"MSA {msa_index} must contain at least one sequence.")
        for sequence_index, sequence in enumerate(msa):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append([rc.HHBLITS_AA_TO_ID[res] for res in sequence])
            deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

    num_res = len(msas[0][0])
    num_alignments = len(int_msa)

    msa_features = {}
    msa_features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
    msa_features["msa"] = np.array(int_msa, dtype=np.int32)
    msa_features["num_alignments"] = np.array([num_alignments] * num_res, dtype=np.int32)

    if use_identifiers:
        msa_features["msa_identifiers"] = descriptions

    return msa_features
