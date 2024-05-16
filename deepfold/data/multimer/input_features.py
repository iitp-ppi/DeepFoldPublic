import collections
import copy
import dataclasses
import itertools
from typing import Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

from deepfold.common import protein
from deepfold.common import residue_constants as rc
from deepfold.data.multimer import msa_pairing
from deepfold.data.search.input_features import create_msa_features
from deepfold.data.search.msa_identifiers import get_uniprot_identifiers

REQUIRED_FEATURES = frozenset(
    {
        "aatype",
        "all_atom_mask",
        "all_atom_positions",
        "all_chains_entity_ids",  #
        "all_crops_all_chains_mask",  #
        "all_crops_all_chains_positions",  #
        "all_crops_all_chains_residue_ids",  #
        "assembly_num_chains",
        "asym_id",
        "bert_mask",
        "cluster_bias_mask",
        "deletion_matrix",
        "deletion_mean",
        "entity_id",
        "entity_mask",
        "mem_peak",
        "msa",
        "msa_mask",
        "num_alignments",
        "num_templates",
        "queue_size",
        "residue_index",
        "resolution",
        "seq_length",
        "seq_mask",
        "sym_id",
        "template_aatype",
        "template_all_atom_mask",
        "template_all_atom_positions",
    }
)

MAX_TEMPLATES = 4
MSA_CROP_SIZE = 2048


def convert_monomer_features(
    monomer_features: dict,
) -> dict:
    """Reshapes and modifies monomer features for multimer models."""
    converted = {}
    unnecessary_leading_dim_feats = {"sequence", "domain_name", "num_alignments", "seq_length"}
    for feature_name, feature in monomer_features.items():
        if feature_name in unnecessary_leading_dim_feats:
            # asarray ensures it's a np.ndarray.
            feature = np.asarray(feature[0], dtype=feature.dtype)
        elif feature_name == "aatype":
            # The multimer model performs the one-hot operation itself.
            feature = np.argmax(feature, axis=-1).astype(np.int32)
        elif feature_name == "template_aatype":
            feature = np.argmax(feature, axis=-1).astype(np.int32)
            new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
            feature = np.take(new_order_list, feature.astype(np.int32), axis=0)
        elif feature_name == "template_all_atom_masks":
            feature_name = "template_all_atom_mask"
        converted[feature_name] = feature
    return converted


def int_id_to_str_id(num: int) -> str:
    """Encodes a number as a string, using reverse spreadsheet style naming.

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


def process_single_chain(
    chain_features: dict,
    is_homomer_or_monomer: bool,
    a3m_strings_for_paring: Sequence[str] | None = None,
) -> dict:
    """Process a single chain features."""
    if a3m_strings_for_paring is None:
        a3m_strings_for_paring = [""]
    if not is_homomer_or_monomer:
        all_seq_msa_features = create_all_seq_msa_features(
            chain_features["sequence"].item().decode("utf8"),
            a3m_strings_for_paring,
        )
        chain_features.update(all_seq_msa_features)
    return chain_features


def add_assembly_features(
    all_chain_features: MutableMapping[str, dict],
) -> MutableMapping[str, dict]:
    """Add features to distinguish between chains.

    Args:
      all_chain_features: A dictionary which maps chain_id to a dictionary of
        features for each chain.

    Returns:
      all_chain_features: A dictionary which maps strings of the form
        `<seq_id>_<sym_id>` to the corresponding chain features. E.g. two
        chains from a homodimer would have keys A_1 and A_2. Two chains from a
        heterodimer would have keys A_1 and B_1.
    """
    # Group the chains by sequence
    seq_to_entity_id = {}
    grouped_chains = collections.defaultdict(list)
    for chain_id, chain_features in all_chain_features.items():
        seq = chain_features["sequence"].item().decode("utf8")
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains[seq_to_entity_id[seq]].append(chain_features)

    new_all_chain_features = {}
    chain_id = 1
    for entity_id, group_chain_features in grouped_chains.items():
        for sym_id, chain_features in enumerate(group_chain_features, start=1):
            new_all_chain_features[f"{int_id_to_str_id(entity_id)}_{sym_id}"] = chain_features
            seq_length = chain_features["seq_length"]
            chain_features["asym_id"] = chain_id * np.ones(seq_length)
            chain_features["sym_id"] = sym_id * np.ones(seq_length)
            chain_features["entity_id"] = entity_id * np.ones(seq_length)
            chain_id += 1

    return new_all_chain_features


def pad_msa(example: dict, min_num_cluster) -> dict:
    example = dict(example)
    num_seq = example["msa"].shape[0]
    if num_seq < min_num_cluster:
        for feat in ("msa", "deletion_matrix", "bert_mask", "msa_mask"):
            example[feat] = np.pad(example[feat], ((0, min_num_cluster - num_seq), (0, 0)))
        example["cluster_bias_mask"] = np.pad(example["cluster_bias_mask"], ((0, min_num_cluster - num_seq),))
    return example


def create_all_seq_msa_features(
    sequence: str,
    a3m_strings: Sequence[str],
) -> dict:
    """Get MSA features for paring."""

    all_seq_features = create_msa_features(
        sequence,
        a3m_strings,
        use_identifiers=True,
    )

    species_id = [get_uniprot_identifiers(s) if s else "" for s in all_seq_features["msa_identifiers"]]
    all_seq_features["msa_identifiers"] = np.array(species_id, dtype=np.object_)

    valid_feats = (*msa_pairing.MSA_FEATURES, "msa_identifiers")
    feats = {f"{k}_all_seq": v for k, v in all_seq_features.items() if k in valid_feats}

    return feats


@dataclasses.dataclass(frozen=False)
class Complex:
    descriptions: List[str] = dataclasses.field(default_factory=list)
    num_units: List[int] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        assert len(self.descriptions) == len(self.num_units)
        assert all(n > 0 for n in self.num_units)


def process(
    complex: Complex,
    all_monomer_features: Mapping[str, dict],
    msas_with_identifiers: Mapping[str, str],
    max_num_clusters: int = 512,
) -> dict:
    """Create a multimer input features."""

    all_chain_features = {}
    is_homomer_or_monomer = len(complex.num_units) == 1

    for cid, desc, num in zip(protein.PDB_CHAIN_IDS, complex.descriptions, complex.num_units):
        assert cid is not None
        chain_features = all_monomer_features[desc]
        chain_features = process_single_chain(
            chain_features,
            is_homomer_or_monomer,
            a3m_strings_for_paring=[msas_with_identifiers[desc]],
        )
        chain_features = convert_monomer_features(chain_features)

        for i in range(num):
            chain_id = f"{cid}_{i+1}"
            all_chain_features[chain_id] = copy.deepcopy(chain_features)

    all_chain_features = add_assembly_features(all_chain_features)

    example = pair_and_merge(all_chain_features)

    example = pad_msa(example, max_num_clusters)

    return example


def _is_homomer_or_monomer(chains: Iterable[dict]) -> bool:
    """Checks if a list of chains represents a homomer/monomer example."""
    # NOTE: An entity_id of 0 indicates padding.
    num_unique_chains = len(np.unique(np.concatenate([np.unique(chain["entity_id"][chain["entity_id"] > 0]) for chain in chains])))
    return num_unique_chains == 1


def pair_and_merge(all_chain_features: MutableMapping[str, dict]) -> dict:
    """Runs processing on features to augment, pair and merge.

    Args:
        all_chain_features: A MutableMap of dictionaries of features for each chain.

    Returns:
        A dictionary of features.

    """
    all_chain_features = process_unmerged_features(all_chain_features)

    np_chains_list = list(all_chain_features.values())

    pair_msa_sequences = not _is_homomer_or_monomer(np_chains_list)

    if pair_msa_sequences:
        np_chains_list = msa_pairing.create_paired_features(chains=np_chains_list)
        np_chains_list = msa_pairing.deduplicate_unpaired_sequences(np_chains_list)
    np_chains_list = crop_chains(
        np_chains_list,
        msa_crop_size=MSA_CROP_SIZE,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=MAX_TEMPLATES,
    )
    np_example = msa_pairing.merge_chain_features(
        np_chains_list=np_chains_list,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=MAX_TEMPLATES,
    )
    np_example = process_final(np_example)
    return np_example


def crop_chains(chains_list: List[dict], msa_crop_size: int, pair_msa_sequences: bool, max_templates: int) -> List[dict]:
    """Crops the MSAs for a set of chains.

    Args:
      chains_list: A list of chains to be cropped.
      msa_crop_size: The total number of sequences to crop from the MSA.
      pair_msa_sequences: Whether we are operating in sequence-pairing mode.
      max_templates: The maximum templates to use per chain.

    Returns:
      The chains cropped.

    """
    # Apply the cropping.
    cropped_chains = []
    for chain in chains_list:
        cropped_chain = _crop_single_chain(
            chain,
            msa_crop_size=msa_crop_size,
            pair_msa_sequences=pair_msa_sequences,
            max_templates=max_templates,
        )
        cropped_chains.append(cropped_chain)

    return cropped_chains


def _crop_single_chain(chain: dict, msa_crop_size: int, pair_msa_sequences: bool, max_templates: int) -> dict:
    """Crops msa sequences to `msa_crop_size`."""
    msa_size = chain["num_alignments"]

    if pair_msa_sequences:
        msa_size_all_seq = chain["num_alignments_all_seq"]
        msa_crop_size_all_seq = np.minimum(msa_size_all_seq, msa_crop_size // 2)

        # We reduce the number of un-paired sequences, by the number of times a
        # sequence from this chain's MSA is included in the paired MSA.  This keeps
        # the MSA size for each chain roughly constant.
        msa_all_seq = chain["msa_all_seq"][:msa_crop_size_all_seq, :]
        num_non_gapped_pairs = np.sum(np.any(msa_all_seq != msa_pairing.MSA_GAP_IDX, axis=1))
        num_non_gapped_pairs = np.minimum(num_non_gapped_pairs, msa_crop_size_all_seq)

        # Restrict the unpaired crop size so that paired+unpaired sequences do not
        # exceed msa_seqs_per_chain for each chain.
        max_msa_crop_size = np.maximum(msa_crop_size - num_non_gapped_pairs, 0)
        msa_crop_size = np.minimum(msa_size, max_msa_crop_size)
    else:
        msa_crop_size = np.minimum(msa_size, msa_crop_size)

    include_templates = "template_aatype" in chain and max_templates
    if include_templates:
        num_templates = chain["template_aatype"].shape[0]
        templates_crop_size = np.minimum(num_templates, max_templates)

    for k in chain:
        k_split = k.split("_all_seq")[0]
        if k_split in msa_pairing.TEMPLATE_FEATURES:
            chain[k] = chain[k][:templates_crop_size, :]
        elif k_split in msa_pairing.MSA_FEATURES:
            if "_all_seq" in k and pair_msa_sequences:
                chain[k] = chain[k][:msa_crop_size_all_seq, :]
            else:
                chain[k] = chain[k][:msa_crop_size, :]

    chain["num_alignments"] = np.asarray(msa_crop_size, dtype=np.int32)
    if include_templates:
        chain["num_templates"] = np.asarray(templates_crop_size, dtype=np.int32)
    if pair_msa_sequences:
        chain["num_alignments_all_seq"] = np.asarray(msa_crop_size_all_seq, dtype=np.int32)
    return chain


def process_final(np_example: dict) -> dict:
    """Final processing steps in data pipeline, after merging and pairing."""
    np_example = _correct_msa_restypes(np_example)
    np_example = _make_seq_mask(np_example)
    np_example = _make_msa_mask(np_example)
    np_example = _filter_features(np_example)
    return np_example


def _correct_msa_restypes(np_example):
    """Correct MSA restype to have the same order as residue_constants."""

    new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE

    np_example["msa"] = np.take(new_order_list, np_example["msa"], axis=0)
    np_example["msa"] = np_example["msa"].astype(np.int32)

    return np_example


def _make_seq_mask(np_example):
    np_example["seq_mask"] = (np_example["entity_id"] > 0).astype(np.float32)
    return np_example


def _make_msa_mask(np_example):
    """Mask features are all ones, but will later be zero-padded."""

    np_example["msa_mask"] = np.ones_like(np_example["msa"], dtype=np.float32)

    seq_mask = (np_example["entity_id"] > 0).astype(np.float32)
    np_example["msa_mask"] *= seq_mask[None]

    return np_example


def _filter_features(np_example: dict) -> dict:
    """Filters features of example to only those requested."""
    return {k: v for (k, v) in np_example.items() if k in REQUIRED_FEATURES}


def process_unmerged_features(all_chain_features: MutableMapping[str, dict]) -> dict:
    """Postprocessing stage for per-chain features before merging."""
    num_chains = len(all_chain_features)
    for chain_features in all_chain_features.values():
        # Convert deletion matrices to float.
        chain_features["deletion_matrix"] = np.asarray(chain_features.pop("deletion_matrix_int"), dtype=np.float32)
        if "deletion_matrix_int_all_seq" in chain_features:
            chain_features["deletion_matrix_all_seq"] = np.asarray(chain_features.pop("deletion_matrix_int_all_seq"), dtype=np.float32)

        chain_features["deletion_mean"] = np.mean(chain_features["deletion_matrix"], axis=0)

        # Add all_atom_mask and dummy all_atom_positions based on aatype.
        all_atom_mask = rc.STANDARD_ATOM_MASK[chain_features["aatype"]]
        chain_features["all_atom_mask"] = all_atom_mask
        chain_features["all_atom_positions"] = np.zeros(list(all_atom_mask.shape) + [3])

        # Add assembly_num_chains.
        chain_features["assembly_num_chains"] = np.asarray(num_chains)

    # Add entity_mask.
    for chain_features in all_chain_features.values():
        chain_features["entity_mask"] = (chain_features["entity_id"] != 0).astype(np.int32)

    return all_chain_features
