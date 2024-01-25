# Copyright 2024 DeepFold Team


"""AlphaFold-Multimer MSA pairing pipeline."""


import collections
from typing import Dict, Iterable, List, Sequence, cast

import numpy as np
import pandas as pd
from scipy import linalg

import deepfold.common.residue_constants as rc
from deepfold.model.v2.data.ops import FeatureDict

# TODO: This stuff should be in a config


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
    "num_sym",  #
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


def create_paired_features(chains: Iterable[FeatureDict]) -> List[FeatureDict]:
    """
    Processes a list of feature dictionaries for protein chains and returns them with paired sequence features.

    This function takes an iterable of feature dictionaries, each representing a protein chain. It pairs the
    sequence features from these chains, ensuring that the sequences are compatible for alignment. The function
    handles padding of features and considers only those rows that are to be paired.

    Args:
        chains (Iterable[FeatureDict]): A list of feature dictionaries for each chain. Each dictionary
                                        contains features related to a protein chain.

    Returns:
        List[FeatureDict]: A list of updated feature dictionaries with sequence features, including only
                           those rows that have been paired.

    Raises:
        ValueError: If any preconditions on the input parameters are violated.
    """

    # Convert the iterable to a list for easier manipulation
    chains = list(chains)

    # Extract the keys from the first chain's feature dictionary
    chain_keys = chains[0].keys()

    # If there are less than 2 chains, return the original chains without modifications
    if len(chains) < 2:
        return chains

    # Process the chains for pairing
    updated_chains = []

    # Pair sequences and get indices for paired rows
    paired_chains_to_paired_row_indices = pair_sequences(chains)
    paired_rows = reorder_paired_rows(paired_chains_to_paired_row_indices)

    # Iterate over each chain and create a new feature dictionary for each
    for chain_num, chain in enumerate(chains):
        # Create a new chain dictionary excluding features ending with '_all_seq'
        new_chain = {k: v for k, v in chain.items() if "_all_seq" not in k}

        # Process each feature that ends with '_all_seq'
        for feature_name in chain_keys:
            if feature_name.endswith("_all_seq"):
                # Pad the features and select only those that are paired
                feats_padded = pad_features(chain[feature_name], feature_name)
                new_chain[feature_name] = feats_padded[paired_rows[:, chain_num]]

        # Add the count of alignments
        new_chain["num_alignments_all_seq"] = np.asarray(len(paired_rows[:, chain_num]))

        # Append the updated chain to the list
        updated_chains.append(new_chain)

    return updated_chains


def pad_features(feature: np.ndarray, feature_name: str) -> np.ndarray:
    """
    Add a 'padding' row at the end of the features list.

    This function is particularly useful in handling cases of partial alignment in
    computational biology, where one chain might not have paired alignment. The padding
    row serves as a placeholder in such scenarios. The type of padding added depends on
    the feature's name. This function asserts that the feature is not of string type
    before proceeding with padding.

    Args:
    feature (np.ndarray): The feature array to be padded.
    feature_name (str): The name of the feature to be padded, which determines
                        the padding strategy.

    Returns:
    np.ndarray: The feature array with an additional padding row added.

    Raises:
    AssertionError: If the feature's data type is a string.
    """

    # Ensure the feature is not of string type
    assert feature.dtype != np.dtype(np.string_)

    # Identify the feature and apply appropriate padding
    if feature_name in ("msa_all_seq", "msa_mask_all_seq", "deletion_matrix_all_seq", "deletion_matrix_int_all_seq"):
        # Determine the number of residues
        num_res = feature.shape[1]
        # Create padding based on predefined values specific to the feature
        padding = MSA_PAD_VALUES[feature_name] * np.ones([1, num_res], feature.dtype)
    elif feature_name == "msa_species_identifiers_all_seq":
        # Special case for species identifiers: use an empty byte string
        padding = [b""]
    else:
        # If the feature does not match any special cases, return it unmodified
        return feature

    # Concatenate the original feature with the padding
    feats_padded = np.concatenate([feature, padding], axis=0)
    return feats_padded


def _make_msa_df(chain_features: FeatureDict) -> pd.DataFrame:
    """
    Constructs a pandas DataFrame containing multiple sequence alignment (MSA) features.

    This function takes a dictionary of chain features and extracts the MSA related data. It calculates
    the per-sequence similarity and gap percentage relative to the query sequence. The output is a
    DataFrame with these computed features along with species identifiers and row indices for each sequence
    in the MSA.

    Args:
        chain_features (FeatureDict): A dictionary containing features of a protein chain,
                                      including MSA and species identifiers.

    Returns:
        pd.DataFrame: A DataFrame with columns for MSA species identifiers, row indices, similarity
                      scores, and gap percentages for each sequence in the MSA.
    """

    # Extract the MSA sequences from the chain features
    chain_msa = chain_features["msa_all_seq"]

    # The first sequence in the MSA is treated as the query sequence
    query_seq = chain_msa[0]

    # Calculate the per-sequence similarity to the query sequence.
    per_seq_similarity = np.sum(query_seq[None] == chain_msa, axis=-1) / float(len(query_seq))

    # Calculate the percentage of gaps in each sequence
    per_seq_gap = np.sum(chain_msa == 21, axis=-1) / float(len(query_seq))

    # Create a DataFrame with MSA species identifiers, row indices, similarity scores, and gap percentages
    msa_df = pd.DataFrame(
        {
            "msa_species_identifiers": chain_features["msa_species_identifiers_all_seq"],
            "msa_row": np.arange(len(chain_features["msa_species_identifiers_all_seq"])),
            "msa_similarity": per_seq_similarity,
            "gap": per_seq_gap,
        }
    )

    return msa_df


def _create_species_dict(msa_df: pd.DataFrame) -> Dict[bytes, pd.DataFrame]:
    """
    Creates a mapping from species identifiers to corresponding DataFrame subsets.

    This function takes a DataFrame containing multiple sequence alignments (MSA)
    and groups them by species identifiers. For each unique species, it creates a
    subset DataFrame. The function returns a dictionary where each key is a species
    identifier (as bytes) and each value is the corresponding subset DataFrame.

    Args:
        msa_df (pd.DataFrame): A DataFrame containing multiple sequence alignments.
            This DataFrame must have a column named 'msa_species_identifiers' which
            contains the species identifiers.

    Returns:
        Dict[bytes, pd.DataFrame]: A dictionary mapping species identifiers to
            their respective subset DataFrames from the MSA DataFrame.
    """

    species_lookup = {}
    # Group the DataFrame by species identifiers and create a dictionary
    for species, species_df in msa_df.groupby("msa_species_identifiers"):
        # Cast the species identifier to bytes and assign the subset DataFrame
        species_lookup[cast(bytes, species)] = species_df

    return species_lookup


def _match_rows_by_sequence_similarity(this_species_msa_dfs: List[pd.DataFrame]) -> List[List[int]]:
    """Finds MSA sequence pairings across chains based on sequence similarity.

    Each chain's MSA sequences are first sorted by their sequence similarity to
    their respective target sequence. The sequences are then paired, starting
    from the sequences most similar to their target sequence.

    Args:
      this_species_msa_dfs: a list of dataframes containing MSA features for
        sequences for a specific species.

    Returns:
     A list of lists, each containing M indices corresponding to paired MSA rows,
     where M is the number of chains.
    """
    all_paired_msa_rows = []

    num_seqs = [len(species_df) for species_df in this_species_msa_dfs if species_df is not None]
    take_num_seqs = np.min(num_seqs)

    sort_by_similarity = lambda x: x.sort_values("msa_similarity", axis=0, ascending=False)

    for species_df in this_species_msa_dfs:
        if species_df is not None:
            species_df_sorted = sort_by_similarity(species_df)
            msa_rows = species_df_sorted.msa_row.iloc[:take_num_seqs].values
        else:
            msa_rows = [-1] * take_num_seqs  # take the last 'padding' row
        all_paired_msa_rows.append(msa_rows)
    all_paired_msa_rows = list(np.array(all_paired_msa_rows).transpose())
    return all_paired_msa_rows


def _match_rows_by_sequence_similarity(this_species_msa_dfs: List[pd.DataFrame]) -> List[List[int]]:
    """
    Finds MSA sequence pairings across chains based on sequence similarity.

    This function takes a list of dataframes, each representing MSA (Multiple Sequence Alignment)
    features for a specific species. It sorts each chain's MSA sequences by their similarity to
    their respective target sequence. Then, it pairs these sequences starting from the ones most
    similar to their target sequences.

    Args:
        this_species_msa_dfs (List[pd.DataFrame]): A list of dataframes containing MSA features for
            sequences for a specific species.

    Returns:
        List[List[int]]: A list of lists, each containing indices corresponding to paired MSA rows.
            These indices represent the row numbers in the MSA dataframes. The length of each list
            is equal to the number of chains.
    """

    # Initialize a list to store the paired MSA row indices
    all_paired_msa_rows = []

    # Determine the minimum number of sequences available across all species
    num_seqs = [len(species_df) for species_df in this_species_msa_dfs if species_df is not None]
    take_num_seqs = np.min(num_seqs)

    # Lambda function to sort dataframes by 'msa_similarity' in descending order
    sort_by_similarity = lambda x: x.sort_values("msa_similarity", axis=0, ascending=False)

    # Iterate over each species dataframe
    for species_df in this_species_msa_dfs:
        if species_df is not None:
            # Sort the dataframe by similarity
            species_df_sorted = sort_by_similarity(species_df)
            # Extract the top 'take_num_seqs' MSA row indices
            msa_rows = species_df_sorted.msa_row.iloc[:take_num_seqs].values
        else:
            # If no data is present, use -1 as a placeholder for missing rows
            msa_rows = [-1] * take_num_seqs

        # Append the MSA row indices for this species to the overall list
        all_paired_msa_rows.append(msa_rows)

    # Transpose the list of lists to align paired rows and return the result
    all_paired_msa_rows = list(np.array(all_paired_msa_rows).transpose())
    return all_paired_msa_rows


def pair_sequences(examples: List[FeatureDict]) -> Dict[int, np.ndarray]:
    """
    Pairs multiple sequence alignments (MSA) across different protein chains based on common species.

    This function processes a list of feature dictionaries, each representing a protein chain.
    It pairs sequences from different chains if they are from the same species, excluding sequences with no specified species.

    Args:
        examples (List[FeatureDict]): A list of feature dictionaries. Each dictionary represents MSA features of a protein chain.

    Returns:
        Dict[int, np.ndarray]: A dictionary where keys are the number of chains and values are numpy arrays of paired MSA rows.
    """

    # Number of examples (protein chains) in the input
    num_examples = len(examples)

    # List to store species dictionaries for each chain
    all_chain_species_dict = []
    # Set to store all common species across chains
    common_species = set()

    # Process each chain to extract species information
    for chain_features in examples:
        msa_df = _make_msa_df(chain_features)  # Convert chain features into a dataframe
        species_dict = _create_species_dict(msa_df)  # Create a species dictionary from the MSA dataframe
        all_chain_species_dict.append(species_dict)
        common_species.update(set(species_dict))

    # Sort and clean the common species set
    common_species = sorted(common_species)
    common_species.remove(b"")  # Remove target sequence species

    # Initialize the dictionary to store paired MSA rows
    all_paired_msa_rows_dict = {k: [] for k in range(num_examples)}
    all_paired_msa_rows_dict[num_examples] = [np.zeros(len(examples), int)]

    # Iterate over each species to find and pair MSA rows
    for species in common_species:
        if not species:
            continue
        this_species_msa_dfs = []
        species_dfs_present = 0
        for species_dict in all_chain_species_dict:
            if species in species_dict:
                this_species_msa_dfs.append(species_dict[species])
                species_dfs_present += 1
            else:
                this_species_msa_dfs.append(None)

        # Skip species present in only one chain
        if species_dfs_present <= 1:
            continue

        # Skip species with any chain having more than 600 rows
        if np.any(
            np.array([len(species_df) for species_df in this_species_msa_dfs if isinstance(species_df, pd.DataFrame)])
            > 600
        ):
            continue

        # Pair MSA rows for the current species
        paired_msa_rows = _match_rows_by_sequence_similarity(this_species_msa_dfs)
        all_paired_msa_rows_dict[species_dfs_present].extend(paired_msa_rows)

    # Convert lists to numpy arrays in the dictionary
    all_paired_msa_rows_dict = {
        num_examples: np.array(paired_msa_rows) for num_examples, paired_msa_rows in all_paired_msa_rows_dict.items()
    }

    return all_paired_msa_rows_dict


def reorder_paired_rows(all_paired_msa_rows_dict: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Creates a sorted list of paired MSA rows.

    This function processes a dictionary where keys represent the number of chains
    involved in a paired alignment and values are arrays of paired indices. It
    orders the paired indices primarily based on the number of chains in the paired
    alignment (higher first) and secondarily based on the product of indices (lower
    e-values first).

    Args:
      all_paired_msa_rows_dict: A dictionary mapping from the number of paired chains
        to the paired indices in the MSA. Key is the number of chains, value is a
        NumPy array of paired indices.

    Returns:
      A NumPy array containing lists of indices. Each list represents a set of paired
      MSA rows, ordered according to the specified criteria.
    """

    all_paired_msa_rows = []

    for num_pairings in sorted(all_paired_msa_rows_dict, reverse=True):
        paired_rows = all_paired_msa_rows_dict[num_pairings]
        paired_rows_product = abs(np.array([np.prod(rows) for rows in paired_rows]))
        paired_rows_sort_index = np.argsort(paired_rows_product)
        all_paired_msa_rows.extend(paired_rows[paired_rows_sort_index])

    return np.array(all_paired_msa_rows)


def block_diag(*arrs: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    """
    Create a block diagonal matrix from the given arrays.

    This function is similar to scipy.linalg.block_diag but adds the capability to
    set a custom padding value for off-diagonal elements.

    Args:
        *arrs: A variable number of numpy arrays. Each array represents a block in the
               resulting block diagonal matrix.
        pad_value: A float representing the padding value for off-diagonal elements.
                   Defaults to 0.0.

    Returns:
        A numpy array representing the block diagonal matrix. Off-diagonal elements
        are set to the specified `pad_value`.

    Example:
        >>> arr1 = np.array([[1, 2], [3, 4]])
        >>> arr2 = np.array([[5, 6]])
        >>> block_diag(arr1, arr2, pad_value=9.0)
        array([[1., 2., 9., 9.],
               [3., 4., 9., 9.],
               [9., 9., 5., 6.]])
    """
    # Create an array of ones with the same shape as each input array.
    ones_arrs = [np.ones_like(x) for x in arrs]

    # Generate a mask for off-diagonal elements (where off-diagonals are 1).
    off_diag_mask = 1.0 - linalg.block_diag(*ones_arrs)

    # Create the block diagonal matrix from input arrays.
    diag = linalg.block_diag(*arrs)

    # Adjust off-diagonal elements to the specified pad value.
    diag += (off_diag_mask * pad_value).astype(diag.dtype)

    return diag


def _correct_post_merged_feats(
    np_example: FeatureDict,
    np_chains_list: Sequence[FeatureDict],
    pair_msa_sequences: bool,
) -> FeatureDict:
    """
    Adds features that need to be computed or recomputed after merging sequences.

    Args:
        np_example (FeatureDict): A dictionary containing features of the example.
        np_chains_list (Sequence[FeatureDict]): A sequence of dictionaries, each containing
            features of a chain in the merged sequence.
        pair_msa_sequences (bool): A flag to indicate whether to pair MSA (Multiple Sequence Alignment) sequences.

    Returns:
        FeatureDict: The updated example feature dictionary with computed or recomputed features.

    This function updates the `np_example` dictionary with new or modified features. These include
    sequence length, number of alignments, cluster bias mask, and bert mask. The behavior of the
    function changes based on the `pair_msa_sequences` flag.
    """

    # Update sequence length and number of alignments in np_example
    np_example["seq_length"] = np.asarray(np_example["aatype"].shape[0], dtype=np.int32)
    np_example["num_alignments"] = np.asarray(np_example["msa"].shape[0], dtype=np.int32)

    if not pair_msa_sequences:
        # Generate cluster bias masks for non-paired MSA sequences
        # cluster_bias_masks = []
        for chain in np_chains_list:
            mask = np.zeros(chain["msa"].shape[0])
            mask[0] = 1  # Set the first row to 1 to include the query sequence
            # cluster_bias_masks.append(mask)
        # np_example["cluster_bias_mask"] = np.concatenate(cluster_bias_masks)

        # Initialize BERT mask for MSA sequences
        msa_masks = [np.ones(x["msa"].shape, dtype=np.float32) for x in np_chains_list]
        np_example["bert_mask"] = block_diag(*msa_masks, pad_value=0)
    else:
        # Handle paired MSA sequences
        # np_example["cluster_bias_mask"] = np.zeros(np_example["msa"].shape[0])
        # np_example["cluster_bias_mask"][0] = 1

        # Initialize BERT masks for paired MSA sequences
        msa_masks = [np.ones(x["msa"].shape, dtype=np.float32) for x in np_chains_list]
        msa_masks_all_seq = [np.ones(x["msa_all_seq"].shape, dtype=np.float32) for x in np_chains_list]

        msa_mask_block_diag = block_diag(*msa_masks, pad_value=0)
        msa_mask_all_seq = np.concatenate(msa_masks_all_seq, axis=1)
        np_example["bert_mask"] = np.concatenate([msa_mask_all_seq, msa_mask_block_diag], axis=0)

    return np_example


def _pad_templates(chains: Sequence[FeatureDict], max_templates: int) -> Sequence[FeatureDict]:
    """
    Pads the template features in each protein chain to a specified maximum number.

    This function iterates over a list of protein chains. For each chain, it checks
    the features that are designated as template features. If the number of template
    features is less than the specified maximum, it pads them with zeros to reach the
    desired count.

    Args:
      chains (Sequence[FeatureDict]): A sequence of dictionaries representing protein chains,
        where each dictionary contains different features of the chain.
      max_templates (int): The desired number of templates each chain should have.

    Returns:
      Sequence[FeatureDict]: The modified sequence of chains, where each chain's template
      features are padded to have a total count equal to `max_templates`.
    """

    for chain in chains:  # Iterate through each chain in the sequence
        for k, v in chain.items():  # Iterate through each key-value pair in the chain
            if k in TEMPLATE_FEATURES:  # Check if the key is one of the template features
                # Calculate the padding required to reach max_templates
                padding = np.zeros_like(v.shape)
                padding[0] = max_templates - v.shape[0]
                padding = [(0, p) for p in padding]  # Format padding as (before, after) for each dimension

                # Pad the feature with zeros to achieve the desired number of templates
                chain[k] = np.pad(v, padding, mode="constant")
    return chains


def _merge_features_from_multiple_chains(chains: Sequence[FeatureDict], pair_msa_sequences: bool) -> FeatureDict:
    """
    Merge features from multiple protein chains.

    This function processes a list of feature dictionaries, each representing a protein chain.
    It merges these features according to specific rules for different feature types: MSA features,
    sequence features, template features, and chain features.

    Args:
        chains (Sequence[FeatureDict]): List of FeatureDict each representing a protein chain. These dictionaries
                contain features like MSA (Multiple Sequence Alignment), sequence features, etc.
        pair_msa_sequences (bool): A boolean flag that determines the merging strategy for MSA features. If True,
                MSA features are concatenated along the num_res dimension. If False, they are block
                diagonalized.

    Returns:
        FeatureDict: A single feature dictionary representing the merged features from all input chains.
    """

    # Initialize a dictionary to hold the merged features
    merged_example = {}

    # Iterate over each feature in the first chain (assuming all chains have the same set of features)
    for feature_name in chains[0]:
        # Extract the same feature across all chains
        feats = [x[feature_name] for x in chains]

        # Split the feature name to identify its type
        feature_name_split = feature_name.split("_all_seq")[0]

        # Process MSA features
        if feature_name_split in MSA_FEATURES:
            # Concatenate MSA features if pair_msa_sequences is True or if feature has "_all_seq"
            if pair_msa_sequences or "_all_seq" in feature_name:
                merged_example[feature_name] = np.concatenate(feats, axis=1)
                # Special handling for 'msa' feature
                if feature_name_split == "msa":
                    merged_example["msa_chains_all_seq"] = np.ones(merged_example[feature_name].shape[0]).reshape(-1, 1)
            else:
                # Use block diagonalization for MSA features otherwise
                merged_example[feature_name] = block_diag(*feats, pad_value=MSA_PAD_VALUES[feature_name])
                # Special handling for 'msa' feature in block diagonalization
                if feature_name_split == "msa":
                    msa_chains = []
                    for i, feat in enumerate(feats):
                        cur_shape = feat.shape[0]
                        vals = np.ones(cur_shape) * (i + 2)
                        msa_chains.append(vals)
                    merged_example["msa_chains"] = np.concatenate(msa_chains).reshape(-1, 1)

        # Process Sequence features
        elif feature_name_split in SEQ_FEATURES:
            merged_example[feature_name] = np.concatenate(feats, axis=0)

        # Process Template features
        elif feature_name_split in TEMPLATE_FEATURES:
            merged_example[feature_name] = np.concatenate(feats, axis=1)

        # Process Chain features
        elif feature_name_split in CHAIN_FEATURES:
            # Sum up the chain features
            merged_example[feature_name] = np.sum(feats).astype(np.int32)

        # For any other type of feature, just use the feature from the first chain
        else:
            merged_example[feature_name] = feats[0]

    return merged_example


def _merge_homomers_dense_msa(chains: Iterable[FeatureDict]) -> Sequence[FeatureDict]:
    """
    Merge all identical chains, making the resulting MSA dense.

    Args:
        chains: An iterable of features for each chain.

    Returns:
        A list of feature dictionaries. All features with the same entity_id
        will be merged - MSA features will be concatenated along the num_res
        dimension - making them dense.
    """
    entity_chains = collections.defaultdict(list)
    for chain in chains:
        entity_id = chain["entity_id"][0]
        entity_chains[entity_id].append(chain)

    grouped_chains = []
    for entity_id in sorted(entity_chains):
        chains = entity_chains[entity_id]
        grouped_chains.append(chains)
    chains = [_merge_features_from_multiple_chains(chains, pair_msa_sequences=True) for chains in grouped_chains]

    return chains


def _concatenate_paired_and_unpaired_features(example: FeatureDict) -> FeatureDict:
    """
    Merges paired and block-diagonalized features.

    Args:
        example (FeatureDict): Merged features.

    Returns:
        FeatureDict: Merged and paired features.
    """

    features = MSA_FEATURES + ("msa_chains",)
    for key in features:
        if key in example:
            feat = example[key]
            feat_all_seq = example[f"{key}_all_seq"]

            try:
                merged_feat = np.concatenate([feat_all_seq, feat], axis=0)
            except Exception as ex:
                raise Exception("Concat failed.", key, feat_all_seq.shape, feat.shape, ex.__class__, ex)

            example[key] = merged_feat
    example["num_alignments"] = np.array(example["msa"].shape[0], dtype=np.int32)

    return example


def merge_chain_features(
    np_chains_list: List[FeatureDict],
    pair_msa_sequences: bool,
    max_templates: int,
) -> FeatureDict:
    """
    Merges features for multiple chains to single FeatureDict.

    Args:
        np_chains_list (List[FeatureDict]): List of FeatureDicts for each chain.
        pair_msa_sequences: Whether to merge paired MSAs.
        max_templates: The maximum number of templates to include.

    Returns:
        FeatureDict: Single FeatureDict for entire complex.
    """

    np_chains_list = _pad_templates(np_chains_list, max_templates)
    np_chains_list = _merge_homomers_dense_msa(np_chains_list)

    # Unpaired MSA features will be always block-diagnolized
    # Paired MSA featuers will be concatenated
    np_example = _merge_features_from_multiple_chains(np_chains_list, pair_msa_sequences=False)
    if pair_msa_sequences:
        np_example = _concatenate_paired_and_unpaired_features(np_example)

    np_example = _correct_post_merged_feats(np_example, np_chains_list, pair_msa_sequences)

    return np_example


def deduplicate_unpaired_sequences(np_chains: List[FeatureDict]) -> List[FeatureDict]:
    """
    Removes unpaired sequences that duplicate a paired sequence within each chain.

    Each 'chain' in the list is a dictionary representing biological sequence data. This function
    ensures that any unpaired sequences (MSA sequences) that duplicate those in the paired sequences
    are removed from each chain in the provided list.

    Args:
        np_chains (List[FeatureDict]): List of FeatureDicts for each chain.

    Returns:
        List[FeatureDict]: List of deduplicated FeatureDicts.

    Note:
        The function modifies the original list of dictionaries in place but also returns it for convenience.
    """

    feature_names = np_chains[0].keys()  # Extract feature names from the first chain
    msa_features = MSA_FEATURES  # Predefined constant containing MSA features
    cache_msa_features = {}  # Cache for storing processed MSA features

    for chain in np_chains:
        entity_id = int(chain["entity_id"][0])  # Extract the entity ID
        if entity_id not in cache_msa_features:
            # Create a set of unique sequences in 'msa_all_seq'
            sequence_set = set(s.tobytes() for s in chain["msa_all_seq"])
            keep_rows = []  # List to keep track of non-duplicate sequence indices

            # Check each sequence in 'msa'; if it's not a duplicate, add its index to keep_rows
            for row_num, seq in enumerate(chain["msa"]):
                if seq.tobytes() not in sequence_set:
                    keep_rows.append(row_num)

            new_msa_features = {}
            for feature_name in feature_names:
                if feature_name in msa_features:
                    if keep_rows:
                        new_msa_features[feature_name] = chain[feature_name][keep_rows]
                    else:
                        # Create a zeroed array with the same shape but no rows
                        new_shape = list(chain[feature_name].shape)
                        new_shape[0] = 0
                        new_msa_features[feature_name] = np.zeros(new_shape, dtype=chain[feature_name].dtype)

            # Cache the newly computed features for the current entity
            cache_msa_features[entity_id] = new_msa_features

        # Update the chain with the cached MSA features
        for feature_name in cache_msa_features[entity_id]:
            chain[feature_name] = cache_msa_features[entity_id][feature_name]

        # Update the number of alignments in the chain
        chain["num_alignments"] = np.array(chain["msa"].shape[0], dtype=np.int32)

    return np_chains
