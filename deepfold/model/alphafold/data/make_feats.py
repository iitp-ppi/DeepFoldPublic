# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited


from typing import Any, Optional, Sequence

import numpy as np

from deepfold.common import protein
from deepfold.common import residue_constants as rc
from deepfold.data import mmcif_parsing, parsers
from deepfold.model.alphafold.data.templates import empty_template_feats, get_custom_template_features
from deepfold.model.alphafold.data.types import FeatureDict


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


def make_sequence_features(sequence: str, description: str, num_res: int) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = rc.sequence_to_onehot(
        sequence=sequence,
        mapping=rc.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array([description.encode("utf-8")], dtype=np.object_)
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array([sequence.encode("utf-8")], dtype=np.object_)
    return features


def make_mmcif_features(mmcif_object: mmcif_parsing.MmcifObject, chain_id: str) -> FeatureDict:
    input_sequence = mmcif_object.chain_to_seqres[chain_id]
    description = "_".join([mmcif_object.file_id, chain_id])
    num_res = len(input_sequence)

    mmcif_feats = {}

    mmcif_feats.update(
        make_sequence_features(
            sequence=input_sequence,
            description=description,
            num_res=num_res,
        )
    )

    all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(mmcif_object=mmcif_object, chain_id=chain_id)
    mmcif_feats["all_atom_positions"] = all_atom_positions
    mmcif_feats["all_atom_mask"] = all_atom_mask
    mmcif_feats["resolution"] = np.array([mmcif_object.header["resolution"]], dtype=np.float32)
    mmcif_feats["release_date"] = np.array([mmcif_object.header["release_date"].encode("utf-8")], dtype=np.object_)
    mmcif_feats["is_distillation"] = np.array(0.0, dtype=np.float32)

    return mmcif_feats


def _aatype_to_str_sequence(aatype):
    return "".join([rc.restypes_with_x[aatype[i]] for i in range(len(aatype))])


def make_protein_features(
    protein_object: protein.Protein,
    description: str,
    _is_distillation: bool = False,
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.0]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(1.0 if _is_distillation else 0.0).astype(np.float32)

    return pdb_feats


def make_pdb_features(
    protein_object: protein.Protein,
    description: str,
    is_distillation: bool = True,
    confidence_threshold: float = 50.0,
) -> FeatureDict:
    pdb_feats = make_protein_features(protein_object, description, _is_distillation=True)

    if is_distillation:
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats


def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix],
) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError("At least one MSA must be provided.")

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
    features = {}
    features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
    features["msa"] = np.array(int_msa, dtype=np.int32)
    features["num_alignments"] = np.array([num_alignments] * num_res, dtype=np.int32)
    return features


# Generate 1-sequence MSA features having only the input sequence
def make_dummy_msa_feats(input_sequence):
    msas = [[input_sequence]]
    deletion_matrices = [[[0 for _ in input_sequence]]]
    msa_features = make_msa_features(
        msas=msas,
        deletion_matrices=deletion_matrices,
    )

    return msa_features


def make_sequence_features_with_custom_template(
    sequence: str, mmcif_path: str, pdb_id: str, chain_id: str, kalign_binary_path: str
) -> FeatureDict:
    """
    process a single fasta file using features derived from a single template rather than an alignment
    """
    num_res = len(sequence)

    sequence_features = make_sequence_features(
        sequence=sequence,
        description=pdb_id,
        num_res=num_res,
    )

    msa_data = [[sequence]]
    deletion_matrix = [[[0 for _ in sequence]]]

    msa_features = make_msa_features(msa_data, deletion_matrix)
    template_features = get_custom_template_features(
        mmcif_path=mmcif_path,
        query_sequence=sequence,
        pdb_id=pdb_id,
        chain_id=chain_id,
        kalign_binary_path=kalign_binary_path,
    )

    return {**sequence_features, **msa_features, **template_features.features}
