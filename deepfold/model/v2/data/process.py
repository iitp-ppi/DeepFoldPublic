# Copyright 2024 DeepFold Team


"""Feature processing logics."""


import collections
from typing import Iterable, List, MutableMapping, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

from deepfold.common import residue_constants as rc
from deepfold.model.v2.data import ops
from deepfold.model.v2.data.msa_paring import baseline as msa_pairing

FeatureDict = MutableMapping[str, np.ndarray]
TensorDict = MutableMapping[str, torch.Tensor]


REQUIRED_FEATURES = frozenset(
    {
        "aatype",
        "all_atom_mask",
        "all_atom_positions",
        "all_chains_entity_ids",
        "all_crops_all_chains_mask",
        "all_crops_all_chains_positions",
        "all_crops_all_chains_residue_ids",
        "assembly_num_chains",
        "asym_id",
        "bert_mask",
        # "cluster_bias_mask",
        "deletion_matrix",
        "deletion_mean",
        "entity_id",
        "entity_mask",
        "mem_peak",
        "msa",
        "msa_mask",
        "num_alignments",
        "num_templates",
        "residue_index",
        "resolution",
        "seq_length",
        "seq_mask",
        "sym_id",
        "template_aatype",
        "template_all_atom_mask",
        "template_all_atom_positions",
        # Uni-Fold
        "asym_len",
        "template_sum_probs",
        "num_sym",
        "msa_chains",
    }
)


MAX_TEMPLATES = 4
MSA_CROP_SIZE = 2048


def nonensembled_fns(common_cfg: DictConfig, mode_cfg: DictConfig):
    """
    Constructs a list of data processing operators for non-ensembled data.

    This function creates a pipeline of data processing operators based on the provided configuration.
    It handles various operations such as random deletion of MSA (multiple sequence alignment),
    casting features, correcting MSA restypes, and more, depending on the features enabled in the
    configuration. It supports different versions and template usage.

    Args:
        common_cfg (DictConfig): A configuration object containing common settings,
                                 such as whether to enable certain features like v2_feature,
                                 use_template, etc.
        mode_cfg (DictConfig): A configuration object containing mode-specific settings,
                               such as settings for random deletion of MSA, maximum templates,
                               and template subsampling.

    Returns:
        List[Callable]: A list of data processing operators configured according to the provided settings.
    """

    enable_v2 = common_cfg.v2_feature
    operators = []

    # Append random delete MSA operator if enabled
    if mode_cfg.random_delete_msa:
        operators.append(ops.random_delete_msa(common_cfg.random_delete_msa))

    # Extend the list with common operators
    operators.extend(
        [
            ops.cast_features,
            ops.correct_msa_restypes,
            ops.squeeze_features,
            ops.randomly_replace_msa_with_unknown(0.0),
            ops.make_sequence_mask,
            ops.make_msa_mask,
        ]
    )

    # Add HHblits profile operator based on version
    operators.append(ops.make_hhblits_profile_v2 if enable_v2 else ops.make_hhblits_profile)

    # Add template-related operators if templates are used
    if common_cfg.use_template:
        operators.extend(
            [
                ops.make_template_mask,
                ops.make_pseudo_beta("template_"),
                ops.crop_templates(
                    max_templates=mode_cfg.max_templates,
                    subsample_templates=mode_cfg.subsample_templates,
                ),
            ]
        )

    # Add torsion angles operator for templates if enabled
    if common_cfg.use_template_torsion_angles:
        operators.append(ops.atom37_to_torsion_angles("template_"))

    # Add operators for creating masks and target features
    operators.append(ops.make_atom14_masks)
    operators.append(ops.make_target_feat)

    return operators


def crop_and_fix_size_fns(common_cfg: DictConfig, mode_cfg: DictConfig, crop_and_fix_size_seed: int):
    """
    Configures a sequence of operations to crop and adjust the size of features based on provided configurations.

    This function generates a list of operators based on the given configurations. It considers whether to reduce MSA
    (Multiple Sequence Alignment) clusters, apply cropping (for both multimers and single proteins), select features,
    and adjust sizes to a fixed dimension.

    Args:
        common_cfg (DictConfig): Common configuration settings used across different modes.
        mode_cfg (DictConfig): Mode-specific configuration settings.
        crop_and_fix_size_seed (int): Seed for random operations in cropping functions.

    Returns:
        List[Callable]: A list of function objects (operators) that perform the specified cropping and size-fixing operations.

    Note:
        The function uses conditional logic to determine the appropriate set of operations to apply based on the
        configurations provided. The operators are intended to be used on protein structure data.
    """

    operators = []

    # Calculate the number of MSA clusters to pad based on the configuration settings
    if common_cfg.reduce_msa_clusters_by_max_templates:
        pad_msa_clusters = mode_cfg.max_msa_clusters - mode_cfg.max_templates
    else:
        pad_msa_clusters = mode_cfg.max_msa_clusters

    # Copy the features from the common configuration
    crop_feats = dict(common_cfg.features)

    # Check if fixed size is enabled in the mode configuration
    if mode_cfg.fixed_size:
        # Check if cropping is enabled.
        if mode_cfg.crop:
            # Determine the cropping function based on whether the data is a multimer or not
            if common_cfg.is_multimer:
                crop_fn = ops.crop_to_size_multimer(
                    crop_size=mode_cfg.crop_size,
                    shape_schema=crop_feats,
                    seed=crop_and_fix_size_seed,
                    spatial_crop_prob=mode_cfg.spatial_crop_prob,
                    ca_ca_threshold=mode_cfg.ca_ca_threshold,
                )
            else:
                crop_fn = ops.crop_to_size_single(
                    crop_size=mode_cfg.crop_size,
                    shape_schema=crop_feats,
                    seed=crop_and_fix_size_seed,
                )
            # Add the crop function to the list of operators
            operators.append(crop_fn)

        # Add the operation to select features as defined in crop_feats
        operators.append(ops.select_feat(crop_feats))

        # Add the operation to adjust the size to fixed dimensions
        operators.append(
            ops.make_fixed_size(
                crop_feats,
                pad_msa_clusters,
                common_cfg.max_extra_msa,
                mode_cfg.crop_size,
                mode_cfg.max_templates,
            )
        )

    # Return the list of configured operators
    return operators


def ensembled_fns(common_cfg: DictConfig, mode_cfg: DictConfig):
    """
    Constructs and returns a list of data transformation operators.

    This function creates an ensemble of data transformation operators based on
    the provided configuration parameters. These operators are used to process
    and transform input pipeline data. The ensemble can be adjusted for different
    modes such as multimer and single-chain predictions, and it supports various
    features including block deletion, MSA sampling, and masked MSA.

    Args:
        common_cfg (DictConfig): A configuration dictionary that contains common
            settings applicable across different modes.
        mode_cfg (DictConfig): A configuration dictionary that contains mode-specific
            settings for the data transformation operators.

    Returns:
        List: A list of data transformation operators according to the specified
        configurations.

    """

    operators = []

    # Check if the mode is multimer or not
    multimer_mode = common_cfg.is_multimer

    # Check for the v2feature flag
    enable_v2 = common_cfg.v2_feature

    # Add block delete MSA operator if applicable and not in multimer mode
    if mode_cfg.block_delete_msa and not multimer_mode:
        operators.append(ops.block_delete_msa(common_cfg.block_delete_msa))

    # Add MSA distillation operator if max distillation MSA clusters are defined
    if "max_distillation_msa_clusters" in mode_cfg:
        operators.append(ops.sample_msa_distillation(mode_cfg.max_distillation_msa_clusters))

    # Calculate max MSA clusters with consideration of max templates
    if common_cfg.reduce_msa_clusters_by_max_templates:
        pad_msa_clusters = mode_cfg.max_msa_clusters - mode_cfg.max_templates
    else:
        pad_msa_clusters = mode_cfg.max_msa_clusters
    max_msa_clusters = pad_msa_clusters
    max_extra_msa = common_cfg.max_extra_msa

    # Ensuring MSA is resampled in recycling
    assert common_cfg.resample_msa_in_recycling
    gumbel_sample = common_cfg.gumbel_sample

    # Add sampling MSA operator
    operators.append(
        ops.sample_msa(
            max_msa_clusters,
            keep_extra=True,
            gumbel_sample=gumbel_sample,
            biased_msa_by_chain=mode_cfg.biased_msa_by_chain,
        )
    )

    # Add masked MSA operator if defined
    if "masked_msa" in common_cfg:
        operators.append(
            ops.make_masked_msa(
                common_cfg.masked_msa,
                mode_cfg.masked_msa_replace_fraction,
                gumbel_sample=gumbel_sample,
                share_mask=mode_cfg.share_mask,
            )
        )

    # Add MSA cluster features operators
    if common_cfg.msa_cluster_features:
        if enable_v2:
            operators.append(ops.nearest_neighbor_clusters_v2())
        else:
            operators.append(ops.nearest_neighbor_clusters())
            operators.append(ops.summarize_clusters)

    # Add MSA feature creation operator
    if enable_v2:
        operators.append(ops.make_msa_feat_v2)
    else:
        operators.append(ops.make_msa_feat)

    # Add extra MSA feature operator
    if max_extra_msa:
        if enable_v2:
            operators.append(ops.make_extra_msa_feat(max_extra_msa))
        else:
            operators.append(ops.crop_extra_msa(max_extra_msa))
    else:
        operators.append(ops.delete_extra_msa)

    return operators


@ops.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x


def pad_then_stack(values: List[torch.Tensor]) -> torch.Tensor:
    if len(values[0].shape) >= 1:
        size = max(v.shape[0] for v in values)
        new_values = []
        for v in values:
            if v.shape[0] < size:
                res = values[0].new_zeros(size, *v.shape[1:])
                res[: v.shape[0], ...] = v
            else:
                res = v
            new_values.append(res)
    else:
        new_values = values
    return torch.stack(new_values, dim=0)


def map_fn(fun, x):
    ensembles = [fun(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = pad_then_stack([dict_i[feat] for dict_i in ensembles])
    return ensembled_dict


def label_transform_fn():
    """Processes for supervised features."""
    return [
        ops.make_atom14_masks,
        ops.make_atom14_positions,
        ops.atom37_to_frames,
        ops.atom37_to_torsion_angles(""),
        ops.make_pseudo_beta(""),
        ops.get_backbone_frames,
        ops.get_chi_angles,
    ]


def process_single_label(label: TensorDict, num_ensemble: Optional[int] = None) -> dict:
    assert "aatype" in label
    assert "all_atom_positions" in label
    assert "all_atom_mask" in label
    label = compose(label_transform_fn())(label)
    if num_ensemble is not None:
        label = {k: torch.stack([v for _ in range(num_ensemble)]) for k, v in label.items()}
    return label


def process_labels(labels_list: List[TensorDict], num_ensemble: Optional[int] = None):
    return [process_single_label(l, num_ensemble) for l in labels_list]


def process_features(
    tensors: TensorDict,
    common_cfg: DictConfig,
    mode_cfg: DictConfig,
) -> TensorDict:
    """
    Apply filters and transformations to the data based on the configuration settings.

    This function processes input tensors by applying a series of transformations and filters.
    These operations are determined by the configurations provided in `common_cfg` and `mode_cfg`.
    It supports both single and multi-ensemble modes and includes conditional logic for data distillation.

    Args:
        tensors (TensorDict): A dictionary containing the tensors to be processed.
        common_cfg (DictConfig): A configuration dictionary with common settings for processing.
        mode_cfg (DictConfig): A configuration dictionary with mode-specific settings for processing.

    Returns:
        TensorDict: The processed tensors after applying the specified transformations and filters.
    """

    # Determine if the operation is for distillation
    is_distillation = bool(tensors.get("is_distillation", 0))

    # Check if the operation is in multimer mode based on common configuration
    multimer_mode = common_cfg.is_multimer

    # Retrieve the seed for cropping and fixing size
    crop_and_fix_size_seed = int(tensors["crop_and_fix_size_seed"])

    # Define the cropping function based on the provided configurations
    crop_fn = crop_and_fix_size_fns(common_cfg, mode_cfg, crop_and_fix_size_seed)

    def wrap_ensemble_fn(data: TensorDict, i: int):
        """
        Apply a series of functions over the ensemble dimension.

        This inner function processes the data by applying ensemble-specific functions
        and then crops and selects features based on the mode (multimer or not) and
        whether distillation is being performed.

        Args:
            data (TensorDict): The input data for ensemble processing.
            i (int): The index in the ensemble dimension.

        Returns:
            TensorDict: Processed data after ensemble functions, cropping, and feature selection.
        """
        d = data.copy()

        # Get ensemble functions based on configurations
        fns = ensembled_fns(common_cfg, mode_cfg)
        new_d = compose(fns)(d)

        # Process data based on multimer mode and distillation flag
        if not multimer_mode or is_distillation:
            new_d = ops.select_feat(common_cfg.recycling_features)(new_d)
            return compose(crop_fn)(new_d)
        else:  # Select after crop for spatial cropping
            d = compose(crop_fn)(d)
            d = ops.select_feat(common_cfg.recycling_features)(d)
            return d

    # Get non-ensembled functions based on configurations
    nonensembled = nonensembled_fns(common_cfg, mode_cfg)

    # Extend non-ensembled functions with label transform if conditions are met
    if mode_cfg.supervised and (not multimer_mode or is_distillation):
        nonensembled.extend(label_transform_fn())

    # Apply non-ensembled functions to tensors
    tensors = compose(nonensembled)(tensors)

    # Determine the number of recycling iterations and ensembles
    num_recycling = int(tensors["num_recycling_iters"]) + 1
    num_ensembles = mode_cfg.num_ensembles

    # Apply ensemble processing to tensors
    ensemble_tensors = map_fn(lambda x: wrap_ensemble_fn(tensors, x), torch.arange(num_recycling * num_ensembles))
    tensors = compose(crop_fn)(tensors)

    # Add a dummy dimension to align with recycling features
    tensors = {k: torch.stack([tensors[k]], dim=0) for k in tensors}

    # Update tensors with ensemble tensors and return
    tensors.update(ensemble_tensors)

    return tensors


### Multimer Processing


def _is_homomer_or_monomer(chains: Iterable[FeatureDict]) -> bool:
    """Checks if a list of chains represents a homomer/monomer example."""

    # Note that an entity_id of 0 indicates padding.
    chains = [np.unique(chain["entity_id"][chain["entity_id"] > 0]) for chain in chains]
    num_unique_chains = len(np.unique(np.concatenate(chains)))

    return num_unique_chains == 1


def pair_and_merge(all_chain_features: MutableMapping[str, FeatureDict]) -> FeatureDict:
    """Runs processing on features to augment, pair and merge."""

    process_unmerged_features_(all_chain_features)

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


def crop_chains(
    chains_list: List[FeatureDict],
    msa_crop_size: int,
    pair_msa_sequences: bool,
    max_templates: int,
) -> List[FeatureDict]:
    """
    Crops the MSAs for a set of chains.

    Args:
        chains_list (List[FeatureDict]): A list of chains to be cropped.
        msa_crop_size (int): The total number of sequences to crop from the MSA.
        pair_msa_sequences (bool): Whether we are operating in sequence-paring mode.
        max_templates (int): The maximum templates to use per chain.

    Return:
        List of the chains cropped.
    """

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


def _crop_single_chain(
    chain: FeatureDict,
    msa_crop_size: int,
    pair_msa_sequences: bool,
    max_templates: int,
) -> FeatureDict:
    """Crops an MSA sequences to `msa_crop_size`."""

    msa_size = chain["num_alignments"]

    if pair_msa_sequences:
        msa_size_all_seq = chain["num_alignments_all_seq"]
        msa_crop_size_all_seq = np.minimum(msa_size_all_seq, msa_crop_size // 2)

        # We reduce the number of un-paired sequences, by the number of times a
        # sequence from this chain's MSA is included in the paired MSA.    This keeps
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


def process_final(np_example: FeatureDict) -> FeatureDict:
    """Final processing steps in data pipeline, after merging and pairing."""
    np_example = _make_seq_mask(np_example)
    np_example = _make_msa_mask(np_example)
    np_example = _filter_features(np_example)
    return np_example


def _make_seq_mask(np_example: FeatureDict) -> FeatureDict:
    np_example["seq_mask"] = (np_example["entity_id"] > 0).astype(np.float32)
    return np_example


def _make_msa_mask(np_example: FeatureDict) -> FeatureDict:
    np_example["msa_mask"] = np.ones_like(np_example["msa"], dtype=np.int8)
    seq_mask = (np_example["entity_id"] > 0).astype(np.int8)
    np_example["msa_mask"] *= seq_mask[None]
    return np_example


def _filter_features(np_example: FeatureDict) -> FeatureDict:
    """Filters features of the example."""
    return {k: v for k, v in np_example.items() if k in REQUIRED_FEATURES}


def process_unmerged_features_(all_chain_features: MutableMapping[str, FeatureDict]) -> None:
    """Post-processing stage for per-chain features before merging."""

    num_chains = len(all_chain_features)
    for chain_features in all_chain_features:
        # Convert deletion matrices to float.
        if "deletion_matrix_int" in chain_features:
            chain_features["deletion_matrix"] = np.asarray(chain_features.pop("deletion_matrix_int"), dtype=np.float32)
        if "deletion_matrix_int_all_seq" in chain_features:
            chain_features["deletion_matrix_all_seq"] = np.asarray(
                chain_features.pop("deletion_matrix_int_all_seq"), dtype=np.float32
            )

        chain_features["deletion_mean"] = np.mean(chain_features["deletion_matrix"], axis=0)

        if "all_atom_positions" not in chain_features:
            # Add all_atom_mask and dummy all_atom_positions based on aatype.
            all_atom_mask = rc.STANDARD_ATOM_MASK[chain_features["aatype"]]
            chain_features["all_atom_mask"] = all_atom_mask
            chain_features["all_atom_positions"] = np.zeros(list(all_atom_mask.shape) + [3])

        # Add assembly_num_chains.
        chain_features["assembly_num_chains"] = np.asarray(num_chains)

    # Add entity_mask.
    for chain_features in all_chain_features:
        chain_features["entity_mask"] = (chain_features["entity_id"] != 0).astype(np.int32)


# TODO: Replace
def empty_template_feats(num_res: int) -> FeatureDict:
    return {
        "template_aatype": np.zeros((0, num_res)).astype(np.int64),
        "template_all_atom_positions": np.zeros((0, num_res, 37, 3)).astype(np.float32),
        "template_sum_probs": np.zeros((0, 1)).astype(np.float32),
        "template_all_atom_mask": np.zeros((0, num_res, 37)).astype(np.float32),
    }


def convert_monomer_features(
    monomer_features: FeatureDict,
    chain_id: str,
) -> FeatureDict:
    """Reshapes and modifies monomer features for multimer models."""

    if monomer_features["template_aatype"].shape[0] == 0:
        monomer_features.update(empty_template_feats(monomer_features["aatype"].shape[0]))

    converted = {}
    converted["auth_chain_id"] = np.asarray(chain_id, dtype=np.object_)
    unnecessary_leading_dim_feats = {
        "sequence",
        "domain_name",
        "num_alignments",
        "seq_length",
    }
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
        elif feature_name == "msa":
            feature = feature.astype(np.uint8)

        if feature_name.endswith("_mask"):
            feature = feature.astype(np.float32)

        converted[feature_name] = feature

    if "deletion_matrix_int" in monomer_features:
        monomer_features["deletion_matrix"] = monomer_features.pop("deletion_matrix_int").astype(np.float32)

    converted.pop("template_sum_probs")  # TODO: Check

    return converted


def int_id_to_str_id(num: int) -> str:
    """
    Encodes a number as a string, using reverse spreadsheet style naming.

    Args:
        num (int): A positive integer.

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


def add_assembly_features(
    all_chain_features: MutableMapping[str, FeatureDict],
) -> MutableMapping[str, FeatureDict]:
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
        seq = str(chain_features["sequence"])
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains[seq_to_entity_id[seq]].append(chain_features)

    new_all_chain_features = {}
    chain_id = 1
    for entity_id, group_chain_features in grouped_chains.items():
        for sym_id, chain_features in enumerate(group_chain_features, start=1):
            new_all_chain_features[f"{int_id_to_str_id(entity_id)}_{sym_id}"] = chain_features
            seq_length = chain_features["seq_length"]
            chain_features["asym_id"] = (chain_id * np.ones(seq_length)).astype(np.int64)
            chain_features["sym_id"] = (sym_id * np.ones(seq_length)).astype(np.int64)
            chain_features["entity_id"] = (entity_id * np.ones(seq_length)).astype(np.int64)
            chain_id += 1

    return new_all_chain_features


def pad_msa(np_example: FeatureDict, min_num_seq: int) -> FeatureDict:
    np_example = dict(np_example)
    num_seq = np_example["msa"].shape[0]
    if num_seq < min_num_seq:
        for feat in ("msa", "deletion_matrix", "bert_mask", "msa_mask", "msa_chains"):
            np_example[feat] = np.pad(np_example[feat], ((0, min_num_seq - num_seq), (0, 0)))
        # np_example["cluster_bias_mask"] = np.pad(np_example["cluster_bias_mask"], ((0, min_num_seq - num_seq),))
    return np_example


def post_process(np_example: FeatureDict) -> FeatureDict:
    np_example = pad_msa(np_example, 512)
    no_dim_keys = [
        "num_alignments",
        "assembly_num_chains",
        "num_templates",
        "seq_length",
        "resolution",
    ]
    for k in no_dim_keys:
        if k in np_example:
            np_example[k] = np_example[k].reshape(-1)
    return np_example


def merge_msas(
    msa: np.ndarray,
    del_mat: np.ndarray,
    new_msa: np.ndarray,
    new_del_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    cur_msa_set = set([tuple(m) for m in msa])
    new_rows = []
    for i, s in enumerate(new_msa):
        if tuple(s) not in cur_msa_set:
            new_rows.append(i)
    ret_msa = np.concatenate([msa, new_msa[new_rows]], axis=0)
    ret_del_mat = np.concatenate([del_mat, new_del_mat[new_rows]], axis=0)
    return ret_msa, ret_del_mat
