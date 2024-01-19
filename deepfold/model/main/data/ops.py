# Copyright 2024 DeepFold Team


import itertools
import logging
from functools import reduce, wraps
from operator import add
from typing import Dict, List, MutableMapping, Optional, Sequence

import numpy as np
import torch
from omegaconf import DictConfig

from deepfold.common import residue_constants as rc
from deepfold.utils.geometry import Rigid, Rotation
from deepfold.utils.random import numpy_seed
from deepfold.utils.tensor_utils import batched_gather, one_hot

FeatureDict = MutableMapping[str, np.ndarray]
TensorDict = MutableMapping[str, torch.Tensor]


logger = logging.getLogger(__name__)

NUM_RES = "NUM_RES"
NUM_MSA_SEQ = "NUM_MSA_SEQ"
NUM_EXTRA_SEQ = "NUM_EXTRA_SEQ"
NUM_TEMPLATES = "NUM_TEMPLATES"

MSA_FEATURE_NAMES: List[str] = [
    "msa",
    "deletion_matrix",
    "msa_mask",
    "msa_row_mask",
    "bert_mask",
    "true_msa",
    "msa_chain",
]


def curry1(f):
    """Supply all arugments but the first."""

    @wraps(f)
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


def cast_for_tensor(protein: TensorDict) -> TensorDict:
    """Keeps all ints as int64 and masks as float32.

    Args:
        protein (TensorDict): Input dictionary containing Torch tensors.

    Returns:
        TensorDict: Output dictionary with the same keys, but with modified data types.
    """

    for k, v in protein.items():
        if k.endswith("_mask"):
            # Convert mask tensors to float32
            protein[k] = v.type(torch.float32)
        elif v.dtype in (torch.int32, torch.uint8, torch.int8):
            # Convert int tensors to int64
            protein[k] = v.type(torch.int64)

    return protein


def make_sequence_mask(protein: TensorDict) -> TensorDict:
    """Creates a sequence mask.

    Args:
        protein (TensorDict): Input dictionary containing Torch tensors.

    Returns:
        TensorDict: Output dictionary with an additional "seq_mask" key containing a float32 tensor.
    """

    protein["seq_mask"] = torch.ones(protein["aatype"].shape, dtype=torch.float32)

    return protein


def make_template_mask(protein: TensorDict) -> TensorDict:
    """Creates a template mask.

    Args:
        protein (TensorDict): Input dictionary containing Torch tensors.

    Returns:
        TensorDict: Output dictionary with an additional "template_mask" key containing a float32 tensor.
    """

    protein["template_mask"] = torch.ones(protein["template_aatype"].shape[0], dtype=torch.float32)

    return protein


def correct_msa_restypes(protein: TensorDict) -> TensorDict:
    """Correct MSA restype to have the same order as `residue_constants`.

    Args:
        protein (TensorDict): Input dictionary containing Torch tensors.

    Returns:
        TensorDict: Output dictionary with the "msa" tensor modified to have the correct order.
    """

    protein["msa"] = protein["msa"].long()
    new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = torch.tensor(new_order_list, dtype=torch.int8).unsqueeze(-1).expand(-1, protein["msa"].size(1))
    protein["msa"] = torch.gather(new_order, 0, protein["msa"]).long()

    return protein


def squeeze_features(protein: TensorDict) -> TensorDict:
    """Remove singleton and repeated dimensions in protein features.

    Args:
        protein (TensorDict): A dictionary containing various protein features as torch tensors.

    Returns:
        TensorDict: A modified dictionary with singleton and repeated dimensions removed from specified keys.
    """

    # Remove singleton dimension in 'aatype' if it exists
    if len(protein["aatype"].shape) == 2:
        protein["aatype"] = torch.argmax(protein["aatype"], dim=-1)

    # Convert 'resolution' to a scalar if it is a tensor with a single value
    if "resolution" in protein and len(protein["resolution"].shape) == 1:
        protein["resolution"] = protein["resolution"][0]

    # Loop through keys and remove singleton dimensions if they exist
    for key in (
        "domain_name",
        "msa",
        "num_alignments",
        "seq_length",
        "sequence",
        "superfamily",
        "deletion_matrix",
        "between_segment_residues",
        "residue_index",
        "template_all_atom_mask",
    ):
        if key in protein and len(protein[key].shape):
            final_dim = protein[key].shape[-1]
            if isinstance(final_dim, int) and final_dim == 1:
                protein[key] = torch.squeeze(protein[key], dim=-1)

    # For 'seq_length' and 'num_alignments', convert to scalars if they are tensors with a single value
    for key in ("seq_length", "num_alignments"):
        if key in protein and len(protein[key].shape):
            protein[key] = protein[key][0]

    return protein


@curry1
def randomly_replace_msa_with_unknown(
    protein: TensorDict,
    replace_proportion: float,
) -> TensorDict:
    """
    Replace a portion of the Multiple Sequence Alignment (MSA) with 'X'.

    Args:
        protein (TensorDict): A dictionary containing tensors, including 'msa' and 'aatype'.
        replace_proportion (float): The proportion of MSA and aatype to replace with 'X'.

    Returns:
        TensorDict: The modified protein dictionary with replaced MSA and aatype.
    """

    if replace_proportion > 0.0:
        # Create a random mask for replacing MSA elements
        msa_mask = np.random.rand(protein["msa"].shape < replace_proportion)
        x_idx, gap_idx = 20, 21

        # Apply the mask to exclude gap_idx values and replace selected elements with 'X'
        msa_mask = torch.logical_and(msa_mask, protein["msa"] != gap_idx)
        protein["msa"] = torch.where(msa_mask, torch.ones_like(protein["msa"]) * x_idx, protein["msa"])

        # Create a random mask for replacing aatype elements
        aatype_mask = np.random.rand(protein["aatype"].shape) < replace_proportion

        # Replace selected aatype elements with 'X'
        protein["aatype"] = torch.where(aatype_mask, torch.ones_like(protein["aatype"]) * x_idx, protein["aatype"])

    return protein


def gumbel_noise(shape: Sequence[int]) -> torch.Tensor:
    """Generate Gumbel Noise of given shape.

    This function generates samples from the Gumbel(0, 1) distribution.

    Args:
        shape (tuple): Shape of the noise to return.

    Returns:
        torch.Tensor: Gumbel noise tensor of the given shape.
    """

    eps = 1e-6

    # Generate uniform noise
    uniform_noise = torch.from_numpy(np.random.uniform(0, 1, shape))

    # Apply the Gumbel transformation to the uniform noise
    gumbel = -torch.log(-torch.log(uniform_noise + eps) + eps)

    return gumbel


def gumbel_max_sample(logits: torch.Tensor) -> torch.Tensor:
    """
    Samples from a probability distribution given by 'logits' using the Gumbel-max trick.

    This function efficiently generates samples from a discrete probability distribution
    defined by the logits. It uses the Gumbel-max trick to approximate the sampling process.

    Args:
        logits (Tensor): Logarithm of probabilities to sample from. These probabilities
            can be unnormalized and represent the log-odds of different outcomes.

    Returns:
        Tensor: A one-hot encoded sample representing the chosen outcome based on the logits.
    """

    # Generate Gumbel noise to introduce randomness into the sampling process.
    z = gumbel_noise(logits.shape)

    # Add the generated noise to the logits and determine the index of the maximum value,
    # which corresponds to the chosen outcome.
    sample = torch.argmax(logits + z, dim=-1)

    return sample


def gumbel_argsort_sample_idx(logits: torch.Tensor) -> torch.Tensor:
    """
    Samples with replacement from a distribution given by 'logits'.

    This function uses the Gumbel trick to implement sampling in an efficient manner.
    For a distribution over k items, this function samples k times without replacement,
    effectively sampling a random permutation with probabilities over the permutations
    derived from the logprobs.

    Args:
        logits (torch.Tensor): Logarithm of probabilities to sample from. Probabilities can be unnormalized.

    Returns:
        torch.Tensor: Sampled indices from logprobs.
    """

    # Generate Gumbel noise with the same shape as logits
    z = gumbel_noise(logits.shape)

    # Add Gumbel noise to logits and perform argsort in descending order
    return torch.argsort(logits + z, dim=-1, descending=True)


def uniform_permutation(
    num_seq: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generates a random permutation of integers from 1 to num_seq (inclusive)
    and adds 0 at the beginning of the permutation.

    Args:
        num_seq (int): The number of integers to include in the permutation (excluding 0).

    Returns:
        torch.Tensor: A tensor containing a uniform random permutation of integers from 0 to num_seq.
                     The first element is always 0, and the remaining elements are shuffled.
    """

    # Generate a random permutation of integers from 1 to num_seq (inclusive) using numpy
    rng = np.random.default_rng(seed)
    shuffled = torch.from_numpy(rng.permutation(num_seq - 1) + 1)

    # Concatenate 0 to the beginning of the shuffled permutation
    return torch.cat((torch.tensor([0]), shuffled), dim=0)


def gumbel_permutation(
    msa_mask: torch.Tensor,
    msa_chains: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Perform Gumbel-sorted permutation of input based on a mask and optional chain information.

    Args:
        msa_mask (torch.Tensor): A tensor representing the mask of the MSA.
        msa_chains (torch.Tensor, optional): A tensor representing the chains in the MSA.
                                             Defaults to None.

    Returns:
        torch.Tensor: A tensor containing the indices of the shuffled MSA, based on the Gumbel distribution.

    Raises:
        AssertionError: If the input logits tensor is not one-dimensional.

    Notes:
        This function generates a permutation of indices based on the Gumbel-sorted logits. It takes a binary mask
        indicating the presence of data in each row (msa_mask), and an optional tensor of chain information (msa_chains).

        The permutation is performed as follows:
        1. Calculate logits for each row based on the mask.
        2. Skip the first row in the logits and mask. (The first row is mapped to the target sequence.)
        3. If chain information is provided, adjust logits based on chain structure.
        4. Compute the Gumbel-sorted permutation of indices.
        5. Return the permuted indices.

        The function assumes that the input tensors are properly formatted and consistent.
    """

    # Check if there is data in each row
    has_msa = torch.sum(msa_mask.long(), dim=-1) > 0

    # Initialize logits with zeros, and set logits for rows without data to a large negative value
    logits = torch.zeros_like(has_msa, dtype=torch.float32)
    logits[~has_msa] = -1e6

    # Ensure logits has a single dimension (skip batch dimension)
    assert len(logits.shape) == 1

    # Skip the first row for both logits and mask
    logits = logits[1:]
    has_msa = has_msa[1:]

    # Return early if there are no entries
    if logits.shape[0] == 0:
        return torch.tensor([0])

    # Process msa_chains if provided
    if msa_chains is not None:
        # Skip the first row in chain information and reshape it
        msa_chains = msa_chains[1:].reshape(-1)

        # Reset chains without MSA
        msa_chains[~has_msa] = 0

        # Compute statistics for MSA chains
        keys = np.unique(msa_chains)
        num_has_msa = has_msa.sum()
        num_pair = (msa_chains == 1).sum()
        num_unpair = num_has_msa - num_pair
        num_chains = (keys > 1).sum()

        # Adjust logits based on MSA chains statistics
        logits[has_msa] = 1.0 / (num_has_msa + 1e-6)
        logits[~has_msa] = 0
        for k in keys:
            if k > 1:
                cur_mask = msa_chains == k
                cur_cnt = cur_mask.sum()
                if cur_cnt > 0:
                    logits[cur_mask] *= num_unpair / (num_chains * cur_cnt)

        # Apply the Gumbel-sorted permutation to logits
        logits = torch.log(logits + 1e-6)

    # Generate the permuted indices
    shuffled = gumbel_argsort_sample_idx(logits) + 1

    # Add 0 to the beginning of the permutation
    return torch.cat((torch.tensor([0]), shuffled), dim=0)


@curry1
def sample_msa(
    protein: TensorDict,
    max_seq: int,
    keep_extra: bool,
    gumbel_sample: bool = False,
    biased_msa_by_chain: bool = False,
    seed: Optional[int] = None,
) -> TensorDict:
    """
    Sample MSA (Multiple Sequence Alignment) randomly, with the option to store remaining sequences as `extra_*`.

    Args:
        protein (TensorDict): A dictionary containing MSA data, including keys like "msa", "msa_chains", and others.
        max_seq (int): Maximum number of sequences to select.
        keep_extra (bool): Whether to store remaining sequences as "extra_*".
        gumbel_sample (bool, optional): If True, use Gumbel permutation for sampling. Default is False.
        biased_msa_by_chain (bool, optional): If True, use biased MSA sampling by chain if "msa_chains" key is available in protein. Default is False.

    Returns:
        TensorDict: A modified protein dictionary.
    """

    # Get the total number of sequences in the MSA
    num_seq = protein["msa"].shape[0]

    # Calculate the number of sequences to select (limited by max_seq)
    num_sel = min(max_seq, num_seq)

    # Generate a permutation order for sequence selection
    if not gumbel_sample:
        index_order = uniform_permutation(num_seq, seed=seed)
    else:
        # Check if biased MSA sampling by chain is requested and msa_chains key is available
        msa_chains = protein["msa_chains"] if (biased_msa_by_chain and "msa_chains" in protein) else None
        # Use Gumbel permutation for sequence selection
        index_order = gumbel_permutation(protein["msa_mask"], msa_chains, seed=seed)

    # Split the selected and not selected sequences based on num_sel
    num_sel = min(max_seq, num_seq)
    sel_seq, not_sel_seq = torch.split(index_order, [num_sel, num_seq - num_sel])

    # Iterate through MSA feature names and update the selected sequences
    for k in MSA_FEATURE_NAMES:
        if k in protein:
            # Store remaining sequences as "extra_*" if keep_extra is True
            if keep_extra:
                protein["extra_" + k] = torch.index_select(protein[k], 0, not_sel_seq)
            # Update protein dictionary with selected sequences
            protein[k] = torch.index_select(protein[k], 0, sel_seq)

    return protein


@curry1
def sample_msa_distillation(protein: TensorDict, max_seq: int) -> TensorDict:
    """Sample MSA from the distillation set with the cutoff."""

    if "is_distillation" in protein and protein["is_distillation"] == 1:
        protein = sample_msa(max_seq, keep_extra=False)(protein)

    return protein


@curry1
def random_delete_msa(
    protein: TensorDict,
    config: DictConfig,
) -> TensorDict:
    """
    Randomly deletes sequences from the MSA (Multiple Sequence Alignment) data in the protein dictionary
    to reduce the cost of MSA features.

    Args:
        protein (TensorDict): A dictionary containing tensor data for protein features.
            It should have a key 'msa' that represents the MSA data.
        config (DictConfig): A configuration dictionary that specifies parameters for the deletion process.
            It should have a key 'max_msa_entry' indicating the maximum number of MSA entries to keep.

    Returns:
        TensorDict: The modified protein dictionary with some MSA sequences randomly removed.
    """

    # Get the number of sequences and sequence length in the MSA
    num_seq = protein["msa"].shape[0]
    seq_len = protein["msa"].shape[1]

    # Calculate the maximum number of sequences to keep based on the configuration
    max_seq = config["max_msa_entry"] // seq_len

    # Check if there are more sequences than allowed, and perform random deletion if needed
    if num_seq > max_seq:
        # Generate random indices to keep, ensuring no duplicates
        keep_index = torch.from_numpy(np.random.choice(num_seq - 1, max_seq - 1, replace=False)).long() + 1
        keep_index = torch.sort(keep_index)[0]
        keep_index = torch.cat((torch.tensor([0]), keep_index), dim=0)

        # Iterate through MSA feature names and update them if present in the protein dictionary
        for k in MSA_FEATURE_NAMES:
            if k in protein:
                # Select only the sequences with the chosen indices
                protein[k] = torch.index_select(protein[k], 0, keep_index)

    return protein


def crop_extra_msa(
    protein: TensorDict,
    max_extra_msa: int,
) -> TensorDict:
    """
    MSA features are cropped so only `max_extra_msa` sequences are kept.

    Args:
        protein (TensorDict): Dictionary containing protein data including MSA.
        max_extra_msa (int): Maximum number of sequences to retain in the cropped MSA.

    Returns:
        TensorDict: The modified protein dictionary with cropped extra MSA data.
    """

    # Get the number of sequences in the 'extra_msa' tensor
    num_seq = protein["extra_msa"].shape[0]

    # Determine the number of sequences to select, limited by 'max_extra_msa' or the actual number of sequences
    num_sel = min(max_extra_msa, num_seq)

    # Generate a random permutation of indices and select the first 'num_sel' indices
    select_indices = torch.from_numpy(np.random.permutation(num_seq)[:num_sel])

    # Iterate over the MSA feature names and crop each if it exists in the protein dictionary
    for k in MSA_FEATURE_NAMES:
        if "extra_" + k in protein:
            # Crop the MSA feature using the selected indices
            protein["extra_" + k] = torch.index_select(protein["extra_" + k], 0, select_indices)

    # Return the modified protein dictionary
    return protein


def delete_extra_msa(protein: TensorDict) -> TensorDict:
    """Delete extra MSA features."""

    for k in MSA_FEATURE_NAMES:
        if "extra_" + k in protein:
            del protein["extra_" + k]

    return protein


def block_delete_msa(protein: TensorDict, config: DictConfig) -> TensorDict:
    """
    Sample MSA by deleting contiguous blocks.

    This function takes a protein represented as a dictionary of tensors and a configuration dictionary.
    It samples the Multiple Sequence Alignment (MSA) data by deleting contiguous blocks of sequences.

    Args:
        protein (TensorDict): A dictionary containing protein-related tensors, including the MSA data.
        config (DictConfig): A dictionary containing configuration parameters.

    Returns:
        TensorDict: A modified protein dictionary after block deletion.

    Notes:
        This function is based on the algorithm described in Jumper et al. (2021) Supplementary Algorithm 1 "MSABlockDeletion".
    """

    if "is_distillation" in protein and protein["is_distillation"] == 1:
        # If 'is_distillation' key is present and equals 1, return the input protein as is
        return protein

    num_seq = protein["msa"].shape[0]

    if num_seq <= config.min_num_msa:
        # If the number of sequences is less than or equal to the minimum threshold, return the input protein
        return protein

    # Calculate the number of sequences in each block based on the configured fraction
    block_num_seq = torch.floor(torch.tensor(num_seq, dtype=torch.float32) * config.msa_fraction_per_block).to(
        torch.int32
    )

    if config.randomize_num_blocks:
        # Randomly select the number of blocks if specified in the configuration
        nb = np.random.randint(0, config.num_blocks + 1)
    else:
        # Use the fixed number of blocks from the configuration
        nb = config.num_blocks

    # Generate random starting positions for deletion blocks
    del_block_starts = torch.from_numpy(np.random.randint(0, num_seq, [nb]))

    # Create a tensor representing the positions of sequences to be deleted within blocks
    del_blocks = del_block_starts[:, None] + torch.arange(0, block_num_seq)

    # Ensure that deleted sequence positions are within bounds
    del_blocks = torch.clip(del_blocks, 0, num_seq - 1)

    # Find unique indices of sequences to be deleted
    del_indices = torch.unique(del_blocks.view(-1))

    # Add zeros to ensure cnt_zero > 1
    combined = torch.hstack((torch.arange(0, num_seq)[None], del_indices[None], torch.zeros(2)[None])).long()

    # Find unique indices with their counts
    uniques, counts = combined.unique(return_counts=True)

    # Identify sequences to keep (counts == 1) and sequences to delete (counts > 1)
    difference = uniques[counts == 1]
    intersection = uniques[counts > 1]

    # Extract indices of sequences to keep
    keep_indices = difference.view(-1)

    # Add a zero at the beginning for indexing consistency
    keep_indices = torch.hstack([torch.zeros(1).long()[None], keep_indices[None]]).view(-1)

    # Assert that the first index is 0 (for consistency)
    assert int(keep_indices[0]) == 0

    # Apply the sequence deletion to relevant keys in the protein dictionary
    for key in MSA_FEATURE_NAMES:
        if key in protein:
            protein[key] = torch.index_select(protein[key], 0, index=keep_indices)

    return protein


@curry1
def nearest_neighbor_clusters(
    protein: TensorDict,
    gap_agreement_weight: float = 0.0,
) -> TensorDict:
    """
    Assign each extra MSA sequence to its nearest neighbor in sampled MSA.

    Args:
        protein (TensorDict): A dictionary containing feature tensors.
        gap_agreement_weight (float, optional): Weight for gap agreement term in the assignment calculation. Defaults to 0.0.

    Returns:
        TensorDict: A dictionary containing the input tensors with an additional key:
            - 'extra_cluster_assignment': Tensor containing cluster assignments for extra sequences.
    """

    # Define weight vector with 21 ones for amino acids, one gap_agreement_weight, and one zero for gaps
    weights = torch.cat([torch.ones(21), gap_agreement_weight * torch.ones(1), torch.zeros(1)], 0)

    # Convert the sampled MSA and extra MSA sequences to one-hot encoding
    msa_one_hot = one_hot(protein["msa"], 23)
    sample_one_hot = protein["msa_mask"][:, :, None] * msa_one_hot
    extra_msa_one_hot = one_hot(protein["extra_msa"], 23)
    extra_one_hot = protein["extra_msa_mask"][:, :, None] * extra_msa_one_hot

    num_seq, num_res, _ = sample_one_hot.shape
    extra_num_seq, _, _ = extra_one_hot.shape

    # Compute the matrix multiplication (einsum) between extra_one_hot and sample_one_hot weighted by 'weights'
    a = extra_one_hot.view(extra_num_seq, num_res * 23)
    b = (sample_one_hot * weights).view(num_seq, num_res * 23).transpose(0, 1)
    agreement = a @ b

    # Assign each sequence in the extra sequences to the closest MSA sample
    protein["extra_cluster_assignment"] = torch.argmax(agreement, dim=1).long()

    return protein


def unsorted_segment_sum(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """
    Computes the sum of segments in the input data tensor.

    Args:
        data (torch.Tensor): Input data tensor to be segmented.
        segment_ids (torch.Tensor): Tensor containing segment indices.
        num_segments (int): Number of segments.

    Returns:
        torch.Tensor: Tensor containing the sum of segments.
    """

    # Check that segment_ids is a 1D tensor and has the same number of elements as data
    assert len(segment_ids.shape) == 1 and segment_ids.shape[0] == data.shape[0]

    # Reshape segment_ids to match the shape of data
    segment_ids = segment_ids.view(segment_ids.shape[0], *((1,) * len(data.shape[1:])))

    # Expand segment_ids to match the shape of data
    segment_ids = segment_ids.expand(data.shape)

    # Create a tensor with zeros of shape [num_segments, data.shape[1:]]
    shape = [num_segments] + list(data.shape[1:])

    # Use scatter_add_ to accumulate the values from data into the corresponding segments
    tensor = torch.zeros(*shape).scatter_add_(0, segment_ids, data.float())

    # Cast the result tensor to the same data type as input data
    tensor = tensor.type(data.dtype)

    return tensor


def summarize_clusters(protein: TensorDict) -> TensorDict:
    """
    Summarizes clusters by producing profile and deletion_matrix_mean within each cluster.

    Args:
        protein (TensorDict): A dictionary containing protein data.

    Returns:
        TensorDict: A dictionary containing the protein data with added cluster summary.
            - "cluster_profile" (ndarray): Profile data for each cluster.
            - "cluster_deletion_mean" (ndarray): Mean deletion data for each cluster.
    """

    num_seq = protein["msa"].shape[0]

    def csum(x):
        """
        Computes cumulative sum along the specified axis.

        Args:
            x (ndarray): Input array for cumulative sum.

        Returns:
            ndarray: Cumulative sum of the input array.
        """
        return unsorted_segment_sum(x, protein["extra_cluster_assignment"], num_seq)

    mask = protein["extra_msa_mask"]
    mask_counts = 1e-6 + protein["msa_mask"] + csum(mask)  # Include center

    # TODO: This line is very slow. Need optimization.
    msa_sum = csum(mask[:, :, None] * one_hot(protein["extra_msa"], 23))
    msa_sum += one_hot(protein["msa"], 23)  # Original sequence
    protein["cluster_profile"] = msa_sum / mask_counts[:, :, None]
    del msa_sum

    del_sum = csum(mask * protein["extra_deletion_matrix"])
    del_sum += protein["deletion_matrix"]  # Original sequence
    protein["cluster_deletion_mean"] = del_sum / mask_counts
    del del_sum

    return protein


@curry1
def nearest_neighbor_clusters_v2(
    batch: TensorDict,
    gap_agreement_weight: float = 0.0,
) -> TensorDict:
    """
    Assign each extra MSA sequence to its nearest neighbor in sampled MSA.

    Args:
        batch (TensorDict): A dictionary containing batch data including MSA, masks,
                            deletion matrices, and other necessary information.
        gap_agreement_weight (float, optional): Weight assigned to gap agreement
                                                (default is 0.0).

    Returns:
        TensorDict: A dictionary containing computed cluster profiles and deletion means.

    Notes:
        This function calculates the cluster assignment for extra MSA sequences based on
        their agreement with sampled MSA sequences. It uses a weighted agreement metric,
        down-weighting gap agreement.
        We could use a BLOSUM matrix here.
        Never put weight on agreeing on BERT mask.
    """

    # Calculate weights for agreement, down-weighting gap agreement
    weights = torch.tensor([1.0] * 21 + [gap_agreement_weight] + [0.0], dtype=torch.float32)

    # Extract necessary data from the batch
    msa_mask = batch["msa_mask"]
    extra_mask = batch["extra_msa_mask"]
    msa_one_hot = one_hot(batch["msa"], 23)
    extra_one_hot = one_hot(batch["extra_msa"], 23)

    # Mask the one-hot representations
    msa_one_hot_masked = msa_mask[:, :, None] * msa_one_hot
    extra_one_hot_masked = extra_mask[:, :, None] * extra_one_hot

    # Compute the agreement matrix between MSA sequences and extra sequences
    t1 = weights * msa_one_hot_masked
    t1 = t1.view(t1.shape[0], t1.shape[1] * t1.shape[2])
    t2 = extra_one_hot_masked.view(extra_one_hot.shape[0], extra_one_hot.shape[1] * extra_one_hot.shape[2])
    agreement = torch.einsum("nm,cm->nc", t1, t2)  # t1 @ t2.T

    # Assign clusters based on agreement, applying softmax
    cluster_assignment = torch.nn.functional.softmax(1e3 * agreement, dim=0)
    cluster_assignment *= torch.einsum("mr, nr->mn", msa_mask, extra_mask)

    # Count the number of sequences in each cluster
    cluster_count = torch.sum(cluster_assignment, dim=-1)
    cluster_count += 1.0  # We always include the sequence itself.

    # Compute the cluster profile
    msa_sum = torch.einsum("nm,mrc->nrc", cluster_assignment, extra_one_hot_masked)
    msa_sum += msa_one_hot_masked
    cluster_profile = msa_sum / cluster_count[:, None, None]

    # Compute the cluster deletion mean
    deletion_matrix = batch["deletion_matrix"]
    extra_deletion_matrix = batch["extra_deletion_matrix"]
    del_sum = torch.einsum("nm,mc->nc", cluster_assignment, extra_mask * extra_deletion_matrix)
    del_sum += deletion_matrix  # Original sequence.
    cluster_deletion_mean = del_sum / cluster_count[:, None]

    # Update and return the batch
    batch["cluster_profile"] = cluster_profile
    batch["cluster_deletion_mean"] = cluster_deletion_mean

    return batch


def make_msa_mask(protein: TensorDict) -> TensorDict:
    """
    Creates a mask for multiple sequence alignment (MSA) data.

    Args:
        protein (TensorDict): A dictionary containing protein data.
            It should contain a key "msa" with the MSA data.

    Returns:
        TensorDict: A modified dictionary with two additional keys:
            - "msa_mask": A mask for MSA data with all ones (will be zero-padded later).
            - "msa_row_mask": A mask for rows in MSA data with all ones.

    Note:
        This function initializes the "msa_mask" and "msa_row_mask" keys in the input
        dictionary with appropriate tensor values if they do not exist.
    """

    if "msa_mask" not in protein:
        # Initialize "msa_mask" with all ones of the same shape as "msa"
        protein["msa_mask"] = torch.ones(protein["msa"].shape, dtype=torch.float32)

    # Initialize "msa_row_mask" with all ones for the number of rows in "msa"
    protein["msa_row_mask"] = torch.ones((protein["msa"].shape[0]), dtype=torch.float32)

    return protein


def pseudo_beta_fn(
    aatype: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Create pseudo-beta features.

    This function generates pseudo-beta features for protein structures. It calculates
    the beta-carbon positions for each amino acid based on the amino acid type.

    Args:
        aatype (torch.Tensor): A tensor containing amino acid types.
        all_atom_positions (torch.Tensor): A tensor containing all atom positions.
        all_atom_mask (torch.Tensor, optional): A tensor representing atom masks.

    Returns:
        torch.Tensor: A tensor containing the pseudo-beta features.

    Note:
        The function calculates the pseudo-beta position for each amino acid based on
        its type. If 'all_atom_mask' is provided, it can also calculate a mask for the
        pseudo-beta atoms.
    """

    if aatype.shape[0] > 0:
        is_gly = torch.eq(aatype, rc.restype_order["G"])
        ca_idx = rc.atom_order["CA"]
        cb_idx = rc.atom_order["CB"]
        pseudo_beta = torch.where(
            torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
            all_atom_positions[..., ca_idx, :],
            all_atom_positions[..., cb_idx, :],
        )
    else:
        pseudo_beta = all_atom_positions.new_zeros(*aatype.shape, 3)

    if all_atom_mask is not None:
        if aatype.shape[0] > 0:
            pseudo_beta_mask = torch.where(is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx])
        else:
            pseudo_beta_mask = torch.zeros_like(aatype).float()
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


@curry1
def make_pseudo_beta(
    protein: TensorDict,
    prefix: str = "",
) -> TensorDict:
    """
    Create pseudo-beta (alpha for glycine) position and mask.

    Args:
        protein (TensorDict): A dictionary containing protein data.
        prefix (str, optional): A prefix for variable names, e.g., "template_". Defaults to "".

    Returns:
        TensorDict: A dictionary with pseudo-beta position and mask added.

    Raises:
        AssertionError: Raised if the 'prefix' is not an empty string or "template_".
    """

    assert prefix in ["", "template_"], "Prefix must be an empty string or 'template_'."

    # Call the pseudo_beta_fn function to compute pseudo-beta positions and masks
    (
        protein[prefix + "pseudo_beta"],
        protein[prefix + "pseudo_beta_mask"],
    ) = pseudo_beta_fn(
        protein["template_aatype" if prefix else "aatype"],
        protein[prefix + "all_atom_positions"],
        protein["template_all_atom_mask" if prefix else "all_atom_mask"],
    )

    return protein


def shaped_categorical(
    probs: torch.Tensor,
    seed: Optional[int] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Generate random samples from a categorical distribution.

    Args:
        probs (torch.Tensor): A tensor representing the probabilities of each category.
        seed (Optional[int]): A random seed for reproducibility.
        eps (float, optional): A small value added to probabilities to prevent division by zero.
            Defaults to 1e-10.

    Returns:
        torch.Tensor: A tensor of random samples from the categorical distribution with
        the same shape as the input `probs`, except for the last dimension.
    """

    g = torch.Generator()
    g.manual_seed(seed)

    ps = probs.shape
    num_classes = ps[-1]
    probs = torch.reshape(probs + eps, [-1, num_classes])
    counts = torch.multinomial(probs, 1, generator=g)
    return torch.reshape(counts, ps[:-1])


def make_hhblits_profile(protein: TensorDict) -> TensorDict:
    """
    Compute the HHblits MSA (Multiple Sequence Alignment) profile if not already present.

    Args:
        protein (TensorDict): A dictionary containing protein information, including the MSA.

    Returns:
        TensorDict: The input protein dictionary with the HHblits profile added if not present.

    If the "hhblits_profile" is already present in the input protein dictionary, this function
    returns the unmodified dictionary. Otherwise, it computes the profile for every residue
    over all MSA sequences and adds the result as "hhblits_profile" to the input dictionary.
    """

    if "hhblits_profile" in protein:
        # If the HHblits profile is already computed, return the unmodified input dictionary
        return protein

    # Compute the one-hot encoding of the MSA sequences (22 amino acid types)
    msa_one_hot = one_hot(protein["msa"], 22)

    # Compute the mean of the one-hot encoding along the sequences (dim=0)
    protein["hhblits_profile"] = torch.mean(msa_one_hot, dim=0)

    # Return the input dictionary with the added HHblits profile.
    return protein


def make_msa_profile(
    batch: TensorDict,
    eps: float = 1e-10,
) -> TensorDict:
    """
    Compute the MSA profile.

    Args:
        batch (TensorDict): A dictionary containing input data.

    Returns:
        TensorDict: A dictionary containing the computed MSA profile.

    This function computes the MSA (Multiple Sequence Alignment) profile by processing
    the input data in the 'batch' dictionary. It first one-hot encodes the MSA sequences,
    then applies a mask to select relevant residues, and finally calculates the profile
    by summing over the one-hot encoded values and dividing by the sum of the mask values
    with a small constant added for numerical stability.
    """

    # One-hot encode the MSA sequences (22 classes).
    oh = one_hot(batch["msa"], 22)

    # Create a mask to select relevant residues.
    mask = batch["msa_mask"][:, :, None]

    # Apply the mask to the one-hot encoded sequences.
    oh *= mask

    # Calculate the profile by summing over residues and adding a small constant for stability.
    return oh.sum(dim=0) / (mask.sum(dim=0) + eps)


def make_hhblits_profile_v2(protein: TensorDict) -> TensorDict:
    """
    Compute the HHblits MSA profile if not already present.

    Args:
        protein (TensorDict): A dictionary representing a protein.

    Returns:
        TensorDict: The input protein dictionary with the HHblits MSA profile added if it wasn't already present.
    """

    # Check if the 'hhblits_profile' key is already in the protein dictionary.
    if "hhblits_profile" in protein:
        return protein  # If it's already present, return the unchanged protein dictionary.

    # If 'hhblits_profile' is not present, compute it by calling the 'make_msa_profile' function.
    protein["hhblits_profile"] = make_msa_profile(protein)

    # Return the modified protein dictionary with the 'hhblits_profile' added.
    return protein


def share_mask_by_entity(
    mask_position: torch.Tensor,
    protein: TensorDict,
) -> torch.Tensor:
    """
    Shares a mask with the same entity.

    This function takes a dictionary 'protein' containing information about the protein
    and a tensor 'mask_position' representing the mask positions. It identifies entities
    within the protein and shares the mask for entities with multiple symmetry.

    Args:
        mask_position (torch.Tensor): A tensor representing the mask positions for the protein.
        protein (TensorDict): A dictionary containing information about the protein.
            It should have the following keys:
            - "entity_id": A tensor with entity IDs, which are unique integers for each set of identical chains.
            - "sym_id": A tensor with symmetry IDs, which are unique integers within a set of identical chains.
            - "num_sym": A tensor with the number of chains in each entity.

    Returns:
        torch.Tensor: A tensor with the shared mask positions.

    Note:
        Introduced in Uni-Fold Multimer.
    """

    # Check if 'num_sym' is a key in the protein dictionary.
    # If not, return the original mask_position
    if "num_sym" not in protein:
        return mask_position

    # Extracting entity, symmetry, and number of symmetries from the protein dictionary
    entity_id = protein["entity_id"]
    sym_id = protein["sym_id"]
    num_sym = protein["num_sym"]

    # Getting unique entity IDs
    unique_entity_ids = entity_id.unique()

    # Creating a mask for the first symmetry
    first_sym_mask = sym_id == 1

    # Iterating over each unique entity ID
    for cur_entity_id in unique_entity_ids:
        # Creating a mask for the current entity ID
        cur_entity_mask = entity_id == cur_entity_id

        # Determining the number of symmetries for the current entity
        cur_num_sym = int(num_sym[cur_entity_mask][0])

        # If there are multiple symmetries, apply the first symmetry's mask across all
        if cur_num_sym > 1:
            cur_sym_mask = first_sym_mask & cur_entity_mask
            cur_sym_bert_mask = mask_position[:, cur_sym_mask]

            # Repeating the mask for the number of symmetries and updating mask_position.
            mask_position[:, cur_entity_mask] = cur_sym_bert_mask.repeat(1, cur_num_sym)

    return mask_position


@curry1
def make_masked_msa(
    protein: TensorDict,
    config: DictConfig,
    replace_fraction: float,
    gumbel_sample: bool = False,
    share_mask: bool = False,
    seed: Optional[int] = None,
) -> TensorDict:
    """
    Generates a masked multiple sequence alignment (MSA) for a given protein.

    This function creates a modified version of a protein's MSA where certain
    positions are masked (replaced) based on various probabilities and configurations.
    It's used to training BERT-like models.

    Args:
        protein (TensorDict): A dictionary containing protein information including its MSA and profiles.
        config (DictConfig): A configuration object containing probabilities for different types of masking.
            - 'uniform_prob' (float): The fraction of amino acids in the MSA to be replaced/masked.
            - 'profile_prob' (float): If True, uses Gumbel sampling for selecting replacements. Defaults to False.
            - 'same_prob' (float): If True, the mask is shared across certain entities. Defaults to False.
        replace_fraction (float): The fraction of amino acids in the MSA to be replaced/masked.
        gumbel_sample (bool, optional): If True, uses Gumbel sampling for selecting replacements. Defaults to False.
        share_mask (bool, optional): If True, the mask is shared across certain entities. Defaults to False.
        seed (int, optional): The desired seed.

    Returns:
        TensorDict: The modified protein dictionary with the new MSA and additional mask information.
    """

    # Define a tensor representing a uniform distribution over amino acids
    random_aa = torch.tensor([0.05] * 20 + [0.0, 0.0], dtype=torch.float32)

    # Calculate probabilities for each type of replacement in the MSA.
    categorical_probs = (
        config.uniform_prob * random_aa  # Uniform probability for random amino acids
        + config.profile_prob * protein["hhblits_profile"]  # Probability based on the protein's HHblits profile
        + config.same_prob * one_hot(protein["msa"], 22)  # Probability of keeping the same amino acid
    )

    # Add padding for the [MASK] token, adjusting probabilities accordingly.
    pad_shapes = list(reduce(add, [(0, 0) for _ in range(len(categorical_probs.shape))]))
    pad_shapes[1] = 1  # Adding a new column for the [MASK] token.
    mask_prob = 1.0 - config.profile_prob - config.same_prob - config.uniform_prob  # Probability for masking
    assert mask_prob >= 0.0  # Ensure the mask probability is non-negative.
    categorical_probs = torch.nn.functional.pad(categorical_probs, pad_shapes, value=mask_prob)

    # Determine positions to mask based on the replace_fraction
    sh = protein["msa"].shape
    with numpy_seed(seed, "masked_msa"):
        mask_position = torch.from_numpy(np.random.rand(*sh) < replace_fraction)
    mask_position &= protein["msa_mask"].bool()  # Apply existing mask.

    # Apply additional masking if specified in the protein data
    if "bert_mask" in protein:
        mask_position &= protein["bert_mask"].bool()

    # Share mask across entities if specified
    if share_mask:
        mask_position = share_mask_by_entity(mask_position, protein)

    # Generate the masked MSA using either Gumbel sampling or categorical sampling
    if gumbel_sample:
        logits = torch.log(categorical_probs + 1e-6)
        bert_msa = gumbel_max_sample(logits)
    else:
        bert_msa = shaped_categorical(categorical_probs, seed=seed)

    # Apply the generated mask to the MSA
    bert_msa = torch.where(mask_position, bert_msa, protein["msa"])
    bert_msa *= protein["msa_mask"].long()

    # Update the protein dictionary with the new masked MSA and additional mask information
    protein["bert_mask"] = mask_position.to(torch.float32)
    protein["true_msa"] = protein["msa"]
    protein["msa"] = bert_msa

    return protein


@curry1
def make_fixed_size(
    protein: TensorDict,
    shape_schema: Dict[str, Sequence[Optional[int]]],
    msa_cluster_size: int,
    extra_msa_size: int,
    num_res: int = 0,
    num_templates: int = 0,
):
    """
    Adjusts the size of protein tensors to fixed dimensions.

    This function modifies the input protein dictionary so that its tensors match
    a specified schema in terms of dimensions. It ensures that the dimensions
    of the protein data (e.g., MSA, sequence data) are consistent and match
    the expected size. Padding is added where necessary.

    Args:
        protein (TensorDict): A dictionary containing protein data in tensor form.
        shape_schema (Dict[str, Sequence[Optional[int]]]): A schema defining the
            expected tensor shapes for each key in the protein dictionary.
        msa_cluster_size (int): The size of the MSA cluster.
        extra_msa_size (int): The size of the extra MSA data.
        num_res (int, optional): The number of residues. Defaults to 0.
        num_templates (int, optional): The number of templates to use. Defaults to 0.

    Returns:
        TensorDict: The modified protein dictionary with tensors adjusted to the
        fixed size as per the shape schema.
    """

    def get_pad_size(cur_size, multiplier=4):
        """Calculates the padding size based on the current size and a multiplier.

        Args:
            cur_size (int): The current size of the dimension.
            multiplier (int, optional): The multiplier used for padding calculation. Defaults to 4.

        Returns:
            int: The calculated padding size.
        """
        return max(multiplier, ((cur_size + multiplier - 1) // multiplier) * multiplier)

    # Adjust the number of residues if necessary
    input_num_res = protein["aatype"].shape[0] if "aatype" in protein else protein["msa_mask"].shape[1]
    if input_num_res != num_res:
        num_res = get_pad_size(input_num_res, 4)

    # Adjust the extra MSA size if necessary
    if "extra_msa_mask" in protein:
        input_extra_msa_size = protein["extra_msa_mask"].shape[0]
        if input_extra_msa_size != extra_msa_size:
            extra_msa_size = get_pad_size(input_extra_msa_size, 8)

    # Map of dimensions to pad size
    pad_size_map = {
        NUM_RES: num_res,
        NUM_MSA_SEQ: msa_cluster_size,
        NUM_EXTRA_SEQ: extra_msa_size,
        NUM_TEMPLATES: num_templates,
    }

    # Process each tensor in the protein dictionary
    for k, v in protein.items():
        # Skip certain keys
        if k in ("extra_cluster_assignment"):
            continue

        shape = list(v.shape)
        schema = shape_schema[k]
        msg = "Rank mismatch between shape and shape schema for"
        assert len(shape) == len(schema), f"{msg} {k}: {shape} vs {schema}"

        # Calculate padding size
        pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]

        # Apply padding and reshape
        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        padding.reverse()
        padding = list(itertools.chain(*padding))
        if padding:
            protein[k] = torch.nn.functional.pad(v, padding)
            protein[k] = torch.reshape(protein[k], pad_size)

    return protein


def make_target_feat(protein: TensorDict) -> TensorDict:
    """
    Create target features for a given protein.

    This function processes the input protein data by converting amino acid types to a long data type,
    determining if there are breaks between protein segments, and creating a one-hot encoded matrix
    for amino acid types. The features, including breaks and amino acid type encodings, are then concatenated
    and added back to the protein data.

    Args:
        protein (TensorDict): A dictionary containing protein data. Expected keys are 'aatype', and optionally
                              'between_segment_residues' and 'asym_len'. 'aatype' should be a tensor of amino acid types.

    Returns:
        TensorDict: The updated protein dictionary with a new key 'target_feat', containing the concatenated
                    features (breaks and amino acid type encodings).

    """
    # Convert amino acid types to long data type
    protein["aatype"] = protein["aatype"].long()

    # Determine if there are breaks between segments
    if "between_segment_residues" in protein:
        # Clip values to be between 0 and 1
        has_break = torch.clip(protein["between_segment_residues"].to(torch.float32), 0, 1)
    else:
        # Create a zero tensor if 'between_segment_residues' is not in protein
        has_break = torch.zeros_like(protein["aatype"], dtype=torch.float32)
        if "asym_len" in protein:
            # Compute the ends of entities if 'asym_len' is present
            asym_len = protein["asym_len"]
            entity_ends = torch.cumsum(asym_len, dim=-1)[:-1]
            has_break[entity_ends] = 1.0
        has_break = has_break.float()

    # One-hot encode the amino acid types
    aatype_1hot = one_hot(protein["aatype"], 21)

    # Concatenate the features: breaks and amino acid type encodings
    target_feat = [
        torch.unsqueeze(has_break, dim=-1),
        aatype_1hot,  # Original sequence encoding
    ]

    # Add the concatenated features back to the protein data
    protein["target_feat"] = torch.cat(target_feat, dim=-1)

    return protein


def make_msa_feat(protein: TensorDict) -> TensorDict:
    """
    Create MSA (Multiple Sequence Alignment) features for a given protein.

    This function processes a protein's MSA data and deletion matrix to create a feature tensor.
    It includes the one-hot encoding of the MSA, a binary matrix indicating deletions, and a normalized value of the deletion matrix.
    If available, it also processes cluster profiles and an additional deletion matrix.

    Args:
        protein (TensorDict): A dictionary containing protein data. Expected keys are 'msa', 'deletion_matrix',
                              and optionally 'cluster_profile', 'cluster_deletion_mean', 'extra_deletion_matrix'.

    Returns:
        TensorDict: The input dictionary augmented with the 'msa_feat' key, which contains the concatenated MSA features.
    """

    # One-hot encode the MSA data with 23 categories
    msa_1hot = one_hot(protein["msa"], 23)

    # Create a binary matrix to indicate the presence of a deletion (clipped between 0.0 and 1.0)
    has_deletion = torch.clip(protein["deletion_matrix"], 0.0, 1.0)

    # Normalize the deletion values using arctan transformation
    deletion_value = torch.atan(protein["deletion_matrix"] / 3.0) * (2.0 / np.pi)

    # Initialize the list of MSA features with the one-hot encoded MSA and deletion matrices
    msa_feat = [
        msa_1hot,
        torch.unsqueeze(has_deletion, dim=-1),
        torch.unsqueeze(deletion_value, dim=-1),
    ]

    # If cluster profile data is available, process and add it to the feature list
    if "cluster_profile" in protein:
        deletion_mean_value = torch.atan(protein["cluster_deletion_mean"] / 3.0) * (2.0 / np.pi)
        msa_feat.extend(
            [
                protein["cluster_profile"],
                torch.unsqueeze(deletion_mean_value, dim=-1),
            ]
        )

    # If an extra deletion matrix is available, process it for future use
    if "extra_deletion_matrix" in protein:
        protein["extra_msa_has_deletion"] = torch.clip(protein["extra_deletion_matrix"], 0.0, 1.0)
        protein["extra_msa_deletion_value"] = torch.atan(protein["extra_deletion_matrix"] / 3.0) * (2.0 / np.pi)

    # Concatenate all MSA features along the last dimension and update the protein dictionary
    protein["msa_feat"] = torch.cat(msa_feat, dim=-1)

    return protein


def make_msa_feat_v2(batch: TensorDict) -> TensorDict:
    """
    Create and concatenate multiple sequence alignment (MSA) features.

    This function processes a batch of MSA data, transforming it into various feature
    representations. It includes one-hot encoding of the MSA, deletion matrix processing,
    and concatenation of these features along with cluster profiles and mean deletion values.

    Args:
        batch (TensorDict): A dictionary containing MSA data and related features. Expected keys are
                            'msa', 'deletion_matrix', 'cluster_deletion_mean', and 'cluster_profile'.

    Returns:
        TensorDict: The input dictionary augmented with a new key 'msa_feat', containing the
                    concatenated MSA features.

    """

    # One-hot encode the MSA data.
    msa_1hot = one_hot(batch["msa"], 23)

    # Process the deletion matrix.
    deletion_matrix = batch["deletion_matrix"]
    has_deletion = torch.clip(deletion_matrix, 0.0, 1.0)[..., None]
    deletion_value = (torch.atan(deletion_matrix / 3.0) * (2.0 / np.pi))[..., None]

    # Calculate mean deletion value.
    deletion_mean_value = (torch.arctan(batch["cluster_deletion_mean"] / 3.0) * (2.0 / np.pi))[..., None]

    # Concatenate all MSA features.
    msa_feat = [
        msa_1hot,
        has_deletion,
        deletion_value,
        batch["cluster_profile"],
        deletion_mean_value,
    ]

    # Add the concatenated features to the batch.
    batch["msa_feat"] = torch.concat(msa_feat, dim=-1)

    return batch


@curry1
def make_extra_msa_feat(
    batch: TensorDict,
    num_extra_msa: int,
) -> TensorDict:
    """
    Processes extra MSA (Multiple Sequence Alignment) features for a batch of data.

    This function modifies the batch by updating fields related to extra MSAs. It clips the
    deletion matrix to a range of 0 to 1, calculates the deletion value, and updates the batch
    with these new tensors. The function is specifically designed for handling features related
    to biological sequence data.

    Args:
        batch (TensorDict): A dictionary containing various tensors related to the batch.
                            Key fields include 'extra_msa', 'extra_deletion_matrix', and
                            'extra_msa_mask'.
        num_extra_msa (int): The number of extra MSAs to process from the batch.

    Returns:
        TensorDict: The updated batch with processed extra MSA features.

    Note:
        - 23 in the context refers to 20 standard amino acids, 'X' for unknown amino acids, a gap, and a bert mask.
        - Deletion values are calculated using arctan normalization for better representation.
    """

    # Select the top `num_extra_msa` MSA features from the batch
    extra_msa = batch["extra_msa"][:num_extra_msa]
    deletion_matrix = batch["extra_deletion_matrix"][:num_extra_msa]

    # Clip deletion values between 0 and 1
    has_deletion = torch.clip(deletion_matrix, 0.0, 1.0)

    # Normalize deletion values using arctan function
    deletion_value = torch.atan(deletion_matrix / 3.0) * (2.0 / np.pi)

    # Extract the mask for the selected extra MSAs
    extra_msa_mask = batch["extra_msa_mask"][:num_extra_msa]

    # Update the batch with processed extra MSA features
    batch["extra_msa"] = extra_msa
    batch["extra_msa_mask"] = extra_msa_mask
    batch["extra_msa_has_deletion"] = has_deletion
    batch["extra_msa_deletion_value"] = deletion_value

    return batch


@curry1
def select_feat(
    protein: TensorDict,
    feature_list: Sequence[str],
) -> TensorDict:
    """
    Selects and returns specific features from a given protein.

    This function filters the input protein's features, returning only those
    specified in the feature_list. It is useful for extracting a subset of
    relevant features from a larger set.

    Args:
        protein (TensorDict): A dictionary representing the protein, where keys
                              are feature names and values are their corresponding
                              data.
        feature_list (Sequence[str]): A list of feature names to be selected from
                                      the protein.

    Returns:
        TensorDict: A dictionary containing only the selected features from the
                    original protein.
    """

    return {k: v for k, v in protein.items() if k in feature_list}


def make_atom14_masks(protein: TensorDict) -> TensorDict:
    """
    Construct denser atom positions for a given protein structure.

    This function reduces the dimensions of atom positions from 37 to 14,
    which can be useful for certain types of protein analyses.
    The reduction is achieved through mapping provided by residue constants (rc).

    Args:
        protein (TensorDict): A dictionary containing protein structure information
                              It must include 'aatype' as a key.

    Returns:
        TensorDict: The updated protein dictionary with additional keys representing
                    the mapped atom positions and existence masks for both 14 and 37
                    atom position dimensions.

    Note:
        This function performs a "lazy" operation, meaning if the protein already has
        'atom14_atom_exists' key, it will return the protein as is without modification.
    """

    # Check if 'atom14_atom_exists' is already in protein to avoid redundant computation
    if "atom14_atom_exists" in protein:
        return protein

    # Mapping from 14 atoms to 37 atoms for each residue type
    restype_atom14_to_atom37 = torch.tensor(
        rc.RESTYPE_ATOM14_TO_ATOM37,
        dtype=torch.int64,
        device=protein["aatype"].device,
    )

    # Mapping from 37 atoms to 14 atoms for each residue type
    restype_atom37_to_atom14 = torch.tensor(
        rc.RESTYPE_ATOM37_TO_ATOM14,
        dtype=torch.int64,
        device=protein["aatype"].device,
    )

    # Mask for the existence of 14 atom positions
    restype_atom14_mask = torch.tensor(
        rc.RESTYPE_ATOM14_MASK,
        dtype=torch.float32,
        device=protein["aatype"].device,
    )

    # Mask for the existence of 37 atom positions
    restype_atom37_mask = torch.tensor(rc.restype_atom37_mask, dtype=torch.float32, device=protein["aatype"].device)

    # Converting amino acid types to long for indexing
    protein_aatype = protein["aatype"].long()

    # Mapping indices for 14 to 37 atom position conversion
    protein["residx_atom14_to_atom37"] = restype_atom14_to_atom37[protein_aatype].long()

    # Mapping indices for 37 to 14 atom position conversion
    protein["residx_atom37_to_atom14"] = restype_atom37_to_atom14[protein_aatype].long()

    # Adding existence masks for 14 and 37 atom positions
    protein["atom14_atom_exists"] = restype_atom14_mask[protein_aatype]
    protein["atom37_atom_exists"] = restype_atom37_mask[protein_aatype]

    return protein


def make_atom14_positions(protein: TensorDict) -> TensorDict:
    """
    Constructs denser atom positions in 14 dimensions instead of 37 for a given protein.

    This function manipulates the protein's atom-related data, converting the
    all_atom_positions and associated masks to a 14-dimensional format. It uses
    batched gather operations and tensor transformations to achieve this.

    Args:
        protein (TensorDict): A dictionary containing tensor representations of protein data.
                              Expected keys include 'aatype', 'all_atom_mask', 'all_atom_positions',
                              'atom14_atom_exists', and 'residx_atom14_to_atom37'.

    Returns:
        TensorDict: The modified protein dictionary with updated keys for 14-dimensional atom data.
    """

    # Convert data types of certain protein attributes for consistency
    protein["aatype"] = protein["aatype"].long()
    protein["all_atom_mask"] = protein["all_atom_mask"].float()
    protein["all_atom_positions"] = protein["all_atom_positions"].float()

    # Prepare masks for atom existence and mapping from 14 to 37 dimensions
    residx_atom14_mask = protein["atom14_atom_exists"]
    residx_atom14_to_atom37 = protein["residx_atom14_to_atom37"]

    # Create a mask for known ground truth positions using batched gather
    residx_atom14_gt_mask = residx_atom14_mask * batched_gather(
        protein["all_atom_mask"],
        residx_atom14_to_atom37,
        dim=-1,
        num_batch_dims=len(protein["all_atom_mask"].shape[:-1]),
    )

    # Gather the ground truth positions based on the ground truth mask
    residx_atom14_gt_positions = residx_atom14_gt_mask[..., None] * (
        batched_gather(
            protein["all_atom_positions"],
            residx_atom14_to_atom37,
            dim=-2,
            num_batch_dims=len(protein["all_atom_positions"].shape[:-2]),
        )
    )

    # Update the protein dictionary with new atom data
    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["atom14_gt_exists"] = residx_atom14_gt_mask
    protein["atom14_gt_positions"] = residx_atom14_gt_positions

    # Define transformation matrices and select appropriate matrices for the residue sequence
    renaming_matrices = torch.tensor(
        rc.RENAMING_MATRICES,
        dtype=protein["all_atom_mask"].dtype,
        device=protein["all_atom_mask"].device,
    )
    renaming_transform = renaming_matrices[protein["aatype"]]

    # Apply transformation matrices to ground truth positions
    alternative_gt_positions = torch.einsum("...rac,...rab->...rbc", residx_atom14_gt_positions, renaming_transform)
    protein["atom14_alt_gt_positions"] = alternative_gt_positions

    # Create a mask for alternative ground truth positions
    alternative_gt_mask = torch.einsum("...ra,...rab->...rb", residx_atom14_gt_mask, renaming_transform)
    protein["atom14_alt_gt_exists"] = alternative_gt_mask

    # Define ambiguous mask based on residue type and sequence
    restype_atom14_is_ambiguous = torch.tensor(
        rc.RESTYPE_ATOM14_IS_AMBIGUOUS,
        dtype=protein["all_atom_mask"].dtype,
        device=protein["all_atom_mask"].device,
    )
    protein["atom14_atom_is_ambiguous"] = restype_atom14_is_ambiguous[protein["aatype"]]

    # Return the modified protein dictionary
    return protein


def atom37_to_frames(protein: TensorDict, eps: float = 1e-8) -> TensorDict:
    """
    Converts atom37 representation to rigid frames for each residue in a protein.

    This function processes a protein's atom37 data to compute the ground truth rigid frames
    for each residue. It uses predefined base atom names and atom positions to determine the
    frames. The function also handles ambiguities in the rigid group definitions and adjusts
    the frames accordingly.

    Args:
        protein (TensorDict): A dictionary containing protein data, including amino acid types,
                              all atom positions, and atom masks.
        eps (float, optional): A small epsilon value used for numerical stability. Default is 1e-8.

    Returns:
        TensorDict: The updated protein dictionary containing the ground truth frames, existence
                    masks, and other relevant data for rigid groups in the protein.

    Note:
        The function requires the input protein data to have specific keys such as 'aatype',
        'all_atom_positions', and 'all_atom_mask'.
    """

    # Extract relevant data from the protein dictionary
    aatype = protein["aatype"]
    all_atom_positions = protein["all_atom_positions"]
    all_atom_mask = protein["all_atom_mask"]

    # Determine batch dimensions
    batch_dims = len(aatype.shape[:-1])

    # Initialize base atom names for the rigid groups
    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
    restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]

    # Populate the base atom names based on chi angles
    for restype, restype_letter in enumerate(rc.restypes):
        resname = rc.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if rc.chi_angles_mask[restype][chi_idx]:
                names = rc.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[restype, chi_idx + 4, :] = names[1:]

    # Initialize and populate rigid group masks
    restype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*aatype.shape[:-1], 21, 8),
    )
    restype_rigidgroup_mask[..., 0] = 1
    restype_rigidgroup_mask[..., 3] = 1
    restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(rc.chi_angles_mask)

    # Prepare lookup tables and indices for base atoms
    lookuptable = rc.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom37_idx = lookup(
        restype_rigidgroup_base_atom_names,
    )
    restype_rigidgroup_base_atom37_idx = aatype.new_tensor(
        restype_rigidgroup_base_atom37_idx,
    )
    restype_rigidgroup_base_atom37_idx = restype_rigidgroup_base_atom37_idx.view(
        *((1,) * batch_dims), *restype_rigidgroup_base_atom37_idx.shape
    )

    # Gather base atom positions for each residue
    residx_rigidgroup_base_atom37_idx = batched_gather(
        restype_rigidgroup_base_atom37_idx,
        aatype,
        dim=-3,
        num_batch_dims=batch_dims,
    )

    base_atom_pos = batched_gather(
        all_atom_positions,
        residx_rigidgroup_base_atom37_idx,
        dim=-2,
        num_batch_dims=len(all_atom_positions.shape[:-2]),
    )

    # Calculate ground truth frames using base atom positions
    gt_frames = Rigid.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=eps,
    )

    # Determine which groups exist in each residue
    group_exists = batched_gather(
        restype_rigidgroup_mask,
        aatype,
        dim=-2,
        num_batch_dims=batch_dims,
    )

    # Check for atom existence in the gathered positions
    gt_atoms_exist = batched_gather(
        all_atom_mask,
        residx_rigidgroup_base_atom37_idx,
        dim=-1,
        num_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    # Initialize rotation matrices for adjusting frame orientations
    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 8, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(mat=rots)

    # Apply rotation to the ground truth frames
    gt_frames = gt_frames.compose(Rigid(rots, None))

    # Handle ambiguities in the rigid groups
    restype_rigidgroup_is_ambiguous = all_atom_mask.new_zeros(*((1,) * batch_dims), 21, 8)
    restype_rigidgroup_rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    restype_rigidgroup_rots = torch.tile(
        restype_rigidgroup_rots,
        (*((1,) * batch_dims), 21, 8, 1, 1),
    )

    # Apply rotation to handle ambiguity in the chi angles
    for resname, _ in rc.residue_atom_renaming_swaps.items():
        restype = rc.restype_order[rc.restype_3to1[resname]]
        chi_idx = int(sum(rc.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[..., restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 2, 2] = -1

    residx_rigidgroup_is_ambiguous = batched_gather(
        restype_rigidgroup_is_ambiguous,
        aatype,
        dim=-2,
        num_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = batched_gather(
        restype_rigidgroup_rots,
        aatype,
        dim=-4,
        num_batch_dims=batch_dims,
    )

    # Apply ambiguity rotation to the ground truth frames
    residx_rigidgroup_ambiguity_rot = Rotation(mat=residx_rigidgroup_ambiguity_rot)
    alt_gt_frames = gt_frames.compose(Rigid(residx_rigidgroup_ambiguity_rot, None))

    # Convert frames to tensors for storage
    gt_frames_tensor = gt_frames.to_tensor_4x4()
    alt_gt_frames_tensor = alt_gt_frames.to_tensor_4x4()

    # Update the protein dictionary with the computed data
    protein["rigidgroups_gt_frames"] = gt_frames_tensor
    protein["rigidgroups_gt_exists"] = gt_exists
    protein["rigidgroups_group_exists"] = group_exists
    protein["rigidgroups_group_is_ambiguous"] = residx_rigidgroup_is_ambiguous
    protein["rigidgroups_alt_gt_frames"] = alt_gt_frames_tensor

    return protein


@curry1
def atom37_to_torsion_angles(protein: TensorDict, prefix: str = "") -> TensorDict:
    """
    Converts atom37 representation of a protein to torsion angles.

    This function calculates torsion angles for a protein given its atom37 representation.
    Torsion angles include omega, phi, psi, and chi angles. The function handles empty
    protein structures and calculates angles only for valid amino acid types.

    Args:
        protein (TensorDict): A dictionary containing protein data, including atom positions and types.
        prefix (str): An optional prefix for keys in `protein` dictionary. Defaults to an empty string.

    Returns:
        TensorDict: The input dictionary augmented with torsion angles and their masks.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    # Extract relevant information from the protein TensorDict
    aatype = protein[prefix + "aatype"]
    all_atom_positions = protein[prefix + "all_atom_positions"]
    all_atom_mask = protein[prefix + "all_atom_mask"]

    # Handle empty protein structures
    if aatype.shape[-1] == 0:
        base_shape = aatype.shape
        # Initialize tensors for torsion angles and masks with zeros
        protein[prefix + "torsion_angles_sin_cos"] = all_atom_positions.new_zeros(*base_shape, 7, 2)
        protein[prefix + "alt_torsion_angles_sin_cos"] = all_atom_positions.new_zeros(*base_shape, 7, 2)
        protein[prefix + "torsion_angles_mask"] = all_atom_positions.new_zeros(*base_shape, 7)
        return protein

    # Clamping amino acid types to a maximum value
    aatype = torch.clamp(aatype, max=20)

    # Pre-processing steps for torsion angle calculation
    # Padding and shifting atom positions for torsion angle calculations
    pad = all_atom_positions.new_zeros([*all_atom_positions.shape[:-3], 1, 37, 3])
    prev_all_atom_positions = torch.cat([pad, all_atom_positions[..., :-1, :, :]], dim=-3)

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    # Define atom positions for omega, phi, and psi angles
    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    # Calculate masks for omega, phi, and psi angles
    pre_omega_mask = torch.prod(prev_all_atom_mask[..., 1:3], dim=-1) * torch.prod(all_atom_mask[..., :2], dim=-1)
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
    psi_mask = torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype) * all_atom_mask[..., 4]

    # Process chi angles
    chi_atom_indices = torch.as_tensor(rc.chi_atom_indices, device=aatype.device)
    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2]))

    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)
    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        num_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype)
    chis_mask = chis_mask * chi_angle_atoms_mask

    # Concatenate position data for all torsion angles
    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    # Concatenate masks for all torsion angles
    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    # Calculate torsion angles using Rigid transformations
    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(torsions_atom_pos[..., 3, :])
    torsion_angles_sin_cos = torch.stack([fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1)

    # Normalize torsion angles
    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    # Adjust torsion angles for mirror symmetries
    torsion_angles_sin_cos = (
        torsion_angles_sin_cos
        * all_atom_mask.new_tensor(
            [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
    )

    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(rc.chi_pi_periodic)[aatype, ...]

    mirror_torsion_angles = torch.cat(
        [
            all_atom_mask.new_ones(*aatype.shape, 3),
            1.0 - 2.0 * chi_is_ambiguous,
        ],
        dim=-1,
    )

    alt_torsion_angles_sin_cos = torsion_angles_sin_cos * mirror_torsion_angles[..., None]

    # Adjustments specific to a prefix
    if prefix == "":
        # Use [1, 0] as placeholder for consistent uni-fold structure
        placeholder_torsions = torch.stack(
            [
                torch.ones(torsion_angles_sin_cos.shape[:-1]),
                torch.zeros(torsion_angles_sin_cos.shape[:-1]),
            ],
            dim=-1,
        )
        torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask[..., None] + placeholder_torsions * (
            1 - torsion_angles_mask[..., None]
        )
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask[
            ..., None
        ] + placeholder_torsions * (1 - torsion_angles_mask[..., None])

    # Store the calculated torsion angles and masks back in the protein dictionary
    protein[prefix + "torsion_angles_sin_cos"] = torsion_angles_sin_cos
    protein[prefix + "alt_torsion_angles_sin_cos"] = alt_torsion_angles_sin_cos
    protein[prefix + "torsion_angles_mask"] = torsion_angles_mask

    return protein


def get_backbone_frames(protein: TensorDict) -> TensorDict:
    protein["true_frame_tensor"] = protein["rigidgroups_gt_frames"][..., 0, :, :]
    protein["frame_mask"] = protein["rigidgroups_gt_exists"][..., 0]

    return protein


def get_chi_angles(protein: TensorDict) -> TensorDict:
    dtype = protein["all_atom_mask"].dtype
    protein["chi_angles_sin_cos"] = (protein["torsion_angles_sin_cos"][..., 3:, :]).to(dtype)
    protein["chi_mask"] = protein["torsion_angles_mask"][..., 3:].to(dtype)

    return protein


@curry1
def crop_templates(
    protein: TensorDict,
    max_templates: int,
    subsample_templates: bool = False,
) -> TensorDict:
    """
    Crops templates from the input protein data based on the specified maximum number of templates.

    If `subsample_templates` is True, templates are randomly sampled. Otherwise, the top templates up to
    `max_templates` are selected. The function modifies the input `TensorDict` by updating the template-related
    entries according to the chosen template indices.

    Args:
        protein (TensorDict): A dictionary containing protein data, including template information.
        max_templates (int): The maximum number of templates to retain.
        subsample_templates (bool, optional): Flag to determine if templates should be randomly sampled.
            Defaults to False.

    Returns:
        TensorDict: The modified protein dictionary with the selected number of templates.

    Raises:
        Exception: Catches and prints exceptions related to tensor operations, primarily for debugging.

    Note:
        The function directly modifies the input `protein` dictionary.
    """

    if "template_mask" in protein:
        # Determining the number of templates available in the protein data
        num_templates = protein["template_mask"].shape[-1]
    else:
        num_templates = 0

    # Only sample or select templates if there are any available
    if num_templates > 0:
        if subsample_templates:
            # Sampling templates based on af2's method, with a random upper limit
            max_templates = min(max_templates, np.random.randint(0, num_templates + 1))
            template_idx = torch.tensor(
                np.random.choice(num_templates, max_templates, replace=False),
                dtype=torch.int64,
            )
        else:
            # Selecting the top templates up to the max_templates limit
            template_idx = torch.arange(min(num_templates, max_templates), dtype=torch.int64)
        for k, v in protein.items():
            # Updating template-related entries in the protein dictionary
            if k.startswith("template"):
                try:
                    v = v[template_idx]
                except Exception as ex:
                    # Exception handling for debugging purposes
                    logger.debug(ex.__class__, ex)
                    logger.debug("num_templates", num_templates)
                    logger.debug(k, v.shape)
                    logger.debug("protein:", protein)
                    logger.debug(f"protein_shape: {{k: v.shape for k, v in protein.items() if 'shape' in dir(v)}}")
            protein[k] = v

    return protein


@curry1
def crop_to_size_single(
    protein: TensorDict,
    crop_size: int,
    shape_schema: Dict[str, Sequence[Optional[int]]],
    seed: Optional[int],
) -> TensorDict:
    """
    Crop a protein representation to a specified size.

    This function crops a given protein data structure to a specified size, based on a provided cropping index.
    It's useful for handling protein sequences of varying lengths, especially in the context of machine learning models
    where input size needs to be consistent.

    Args:
        protein (TensorDict): A dictionary containing tensors related to the protein.
            Expected keys include 'aatype' and 'msa_mask'.
        crop_size (int): The size to which the protein representation needs to be cropped.
        shape_schema (Dict[str, Sequence[Optional[int]]]): A schema defining the shape of each tensor
            in the protein dictionary after cropping.
        seed (Optional[int]): An optional seed for random number generation, used in defining the crop index.
            If None, a random seed is used.

    Returns:
        TensorDict: The cropped protein dictionary, with tensors reshaped according to the `shape_schema`.

    """

    # Determine the number of residues based on the 'aatype' key if it exists, otherwise use 'msa_mask'.
    num_res = protein["aatype"].shape[0] if "aatype" in protein else protein["msa_mask"].shape[1]

    # Get the cropping index based on the number of residues, crop size, and optional seed.
    crop_idx = get_single_crop_idx(num_res, crop_size, seed)

    # Apply the crop index to the protein data structure.
    protein = apply_crop_idx(protein, shape_schema, crop_idx)

    return protein


@curry1
def crop_to_size_multimer(
    protein: TensorDict,
    crop_size: int,
    shape_schema: Dict[str, Sequence[Optional[int]]],
    spatial_crop_prob: float,
    ca_ca_threshold: float,
    seed: Optional[int] = None,
) -> TensorDict:
    """
    Crops a multimeric protein sequence to a specified size.

    This function works with multimeric proteins and supports both spatial and contiguous
    cropping. It decides on the cropping strategy based on the provided spatial crop probability.
    For distillation cases, it delegates to cropping a single sequence.

    Args:
        protein (TensorDict): A dictionary containing protein data.
        crop_size (int): The target size to crop the protein sequence to.
        shape_schema (Dict[str, Sequence[Optional[int]]]): Schema defining the shape of the protein data.
        spatial_crop_prob (float): Probability to use spatial cropping.
        ca_ca_threshold (float): Threshold distance for spatial cropping.
        seed (int): Seed for random number generation to ensure reproducibility.

    Returns:
        TensorDict: A protein dictionary cropped to the specified size.

    Raises:
        ValueError: If an invalid cropping strategy is provided.
    """

    with numpy_seed(seed, key="multimer_crop"):
        use_spatial_crop = np.random.rand() < spatial_crop_prob

    is_distillation = "is_distillation" in protein and protein["is_distillation"] == 1
    if is_distillation:
        return crop_to_size_single(crop_size=crop_size, shape_schema=shape_schema, seed=seed)(protein)
    elif use_spatial_crop:
        crop_idx = get_spatial_crop_idx(protein, crop_size, seed, ca_ca_threshold)
    else:
        crop_idx = get_contiguous_crop_idx(protein, crop_size, seed)

    return apply_crop_idx(protein, shape_schema, crop_idx)


def apply_crop_idx(
    protein: TensorDict,
    shape_schema: Dict[str, Sequence[Optional[int]]],
    crop_idx: torch.Tensor,
) -> TensorDict:
    """
    Apply cropping to a protein tensor dictionary based on given indices.

    This function iterates through each key-value pair in the input protein tensor dictionary.
    For each tensor, it checks if its key has a corresponding shape schema. If so, it applies
    indexing on the dimensions where the number of residues (NUM_RES) matches the shape schema,
    using the provided crop indices.

    Args:
        protein (TensorDict): A dictionary where keys are string identifiers for protein data
                               (e.g., 'sequence', 'structure') and values are tensors representing
                               the corresponding data.
        shape_schema (Dict[str, Sequence[Optional[int]]]): A dictionary that defines the expected
                                                           shape of each tensor in the protein dictionary.
                                                           It maps string keys to sequences of integers
                                                           or None, where each integer represents the
                                                           size of a tensor dimension.
        crop_idx (torch.Tensor): A 1D tensor of indices used for cropping the protein tensors.

    Returns:
        TensorDict: A dictionary with the same keys as the input protein dictionary, but with tensors
                    cropped according to the provided indices.
    """

    cropped_protein = {}

    # Iterate over all items in the protein dictionary
    for k, v in protein.items():
        # Skip items that do not have a corresponding shape schema
        if k not in shape_schema:
            continue

        # Apply cropping to the tensor dimensions specified in the shape schema
        for i, dim_size in enumerate(shape_schema[k]):
            if dim_size == NUM_RES:
                # Cropping is applied along the dimension that matches NUM_RES
                v = torch.index_select(v, i, crop_idx)

        # Update the cropped protein dictionary with the modified tensor
        cropped_protein[k] = v

    return cropped_protein


def get_single_crop_idx(
    num_res: int,
    crop_size: int,
    seed: Optional[int],
) -> torch.Tensor:
    """
    Selects a continuous range of indices for cropping a tensor.

    This function generates a range of indices that can be used to crop a tensor. If the number of residues
    (`num_res`) is less than the specified `crop_size`, it returns all indices. Otherwise, it randomly selects a
    starting point and returns a range of indices of length `crop_size` starting from this point.

    Args:
        num_res (int): The number of residues.
        crop_size (int): The size of the crop, i.e., how many consecutive elements should be selected.
        seed (Optional[int]): A random seed, ensuring reproducibility.

    Returns:
        torch.Tensor: A 1D tensor containing the selected range of indices for cropping.

    Raises:
        ValueError: If `num_res` or `crop_size` is negative.
    """
    # Check for valid input values
    if num_res < 0 or crop_size < 0:
        raise ValueError("num_res and crop_size must be non-negative")

    # Return all indices if num_res is less than crop_size
    if num_res < crop_size:
        return torch.arange(num_res)

    # Randomly select a start index for cropping
    with numpy_seed(seed):
        crop_start = crop_start = int(np.random.randint(0, num_res - crop_size + 1))

    # Return the range of indices starting from crop_start
    return torch.arange(crop_start, crop_start + crop_size)


def get_crop_sizes_each_chain(
    asym_len: torch.Tensor,
    crop_size: int,
    seed: Optional[int] = None,
    use_multinomial: bool = False,
) -> torch.Tensor:
    """
    Calculate the crop sizes for each chain in a contiguous crop setting.

    This function computes the crop sizes for each entity (chain) based on their lengths and the total crop size.
    It supports two modes: a default mode that shuffles and allocates crop sizes sequentially, and a multinomial
    mode that allocates based on a probability distribution.

    Args:
        asym_len (torch.Tensor): A tensor containing the lengths of each entity (chain).
        crop_size (int): The total crop size to be distributed among the entities.
        seed (Optional[int], optional): A seed for random number generation to ensure reproducibility. Defaults to None.
        use_multinomial (bool, optional): A flag to use multinomial distribution for crop size allocation. Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the crop sizes allocated to each entity.
    """
    if not use_multinomial:
        # Shuffle indices for random allocation, with optional seed
        with numpy_seed(seed, key="multimer_contiguous_perm"):
            shuffle_idx = np.random.permutation(len(asym_len))

        num_left = asym_len.sum()  # Total remaining length
        num_budget = torch.tensor(crop_size)  # Total crop size budget
        crop_sizes = [0 for _ in asym_len]  # Initialize crop sizes

        # Allocate crop sizes sequentially to each entity
        for j, idx in enumerate(shuffle_idx):
            this_len = asym_len[idx]
            num_left -= this_len
            max_size = min(num_budget, this_len)  # Maximum crop size for this entity
            min_size = min(this_len, max(0, num_budget - num_left))  # Minimum crop size

            # Randomly choose a crop size within the allowed range
            with numpy_seed(seed, j, key="multimer_contiguous_crop_size"):
                this_crop_size = int(np.random.randint(low=int(min_size), high=int(max_size) + 1))

            num_budget -= this_crop_size
            crop_sizes[idx] = this_crop_size

        crop_sizes = torch.tensor(crop_sizes)
    else:
        # Use multinomial distribution for crop size allocation
        entity_probs = asym_len / torch.sum(asym_len)  # Probability distribution based on entity lengths

        with numpy_seed(seed=seed, key="multimer_contiguous_crop_size"):
            crop_sizes = torch.from_numpy(np.random.multinomial(crop_size, pvals=entity_probs))

        # Ensure crop sizes do not exceed the actual entity lengths
        crop_sizes = torch.min(crop_sizes, asym_len)

    return crop_sizes


def get_contiguous_crop_idx(
    protein: TensorDict,
    crop_size: int,
    seed: Optional[int] = None,
    use_multinomial: bool = False,
) -> torch.Tensor:
    """
    Calculate contiguous crop indices for a protein tensor.

    This function determines crop indices for a given protein tensor, ensuring that the
    crops are contiguous within the protein structure. It supports variable chain lengths
    and can optionally use a multinomial distribution for crop size determination.

    Args:
        protein (TensorDict): A dictionary containing protein data with keys like 'aatype'
                              and 'asym_len'. 'aatype' is a tensor indicating amino acid types,
                              and 'asym_len' is a tensor indicating the lengths of asymmetric units.
        crop_size (int): The size of the crop to be taken from the protein.
        seed (Optional[int], optional): A seed for random number generation, ensuring reproducibility.
                                        Defaults to None.
        use_multinomial (bool, optional): Flag to use multinomial distribution for crop size
                                          determination. Defaults to False.

    Returns:
        torch.Tensor: A tensor of indices indicating the start and end of the contiguous crop.

    Raises:
        AssertionError: If 'asym_len' key is not in the protein dictionary.
    """

    # Get the number of residues in the protein
    num_res = protein["aatype"].shape[0]
    # Return early if the protein is smaller or equal to the crop size
    if num_res <= crop_size:
        return torch.arange(num_res)

    # Ensuring 'asym_len' key exists in the protein dictionary
    assert "asym_len" in protein
    asym_len = protein["asym_len"]

    # Calculate the crop sizes for each chain
    crop_sizes = get_crop_sizes_each_chain(asym_len, crop_size, seed, use_multinomial)
    crop_idxs = []
    asym_offset = torch.tensor(0, dtype=torch.int64)

    # Generate crop start indices with an optional seed for reproducibility
    with numpy_seed(seed, key="multimer_contiguous_crop_start_idx"):
        for l, csz in zip(asym_len, crop_sizes):
            # Randomly select a start index for the crop
            this_start = np.random.randint(0, int(l - csz) + 1)
            # Append the range of indices for this crop
            crop_idxs.append(torch.arange(asym_offset + this_start, asym_offset + this_start + csz))
            asym_offset += l

    # Concatenate all crop index ranges into a single tensor
    return torch.concat(crop_idxs)


def get_spatial_crop_idx(
    protein: TensorDict,
    crop_size: int,
    random_seed: int,
    ca_ca_threshold: float,
    inf: float = 3e4,
) -> List[int]:
    """
    Computes indices for spatial cropping of a protein based on C-alpha atom positons.

    This function identifies a subset of C-alpha atoms in a protein that are within a specified
    threshold distance. If suitable interface candidates are found, it selects a random target
    residue and finds the closest residues to this target, up to the specified crop size. If no
    interface candidates are found, a contiguous crop index is returned.

    Args:
        protein (TensorDict): A dictionary containing protein data, including atom positions and masks.
        crop_size (int): The number of residues to include in the crop.
        random_seed (int): Seed for random number generator to ensure reproducibility.
        ca_ca_threshold (float): The threshold distance for considering C-alpha atoms as interface candidates.
        inf (float, optional): A large number used to represent infinity. Defaults to 3e4.

    Returns:
        List[int]: A list of indices representing the crop of the protein around a selected residue.

    Raises:
        ValueError: If `crop_size` is not positive.

    Note:
        This function assumes that the protein data includes 'all_atom_positions' and 'all_atom_mask' keys.
    """

    # Get the index of C-alpha atoms in the protein
    ca_idx = rc.atom_order["CA"]

    # Extract coordinates and mask of C-alpha atoms
    ca_coords = protein["all_atom_positions"][..., ca_idx, :]
    ca_mask = protein["all_atom_mask"][..., ca_idx].bool()

    # Check if there are enough atoms to construct interface; if not, use contiguous crop
    if (ca_mask.sum(dim=-1) <= 1).all():
        return get_contiguous_crop_idx(protein, crop_size, random_seed)

    # Calculate pairwise distance mask for C-alpha atoms
    pair_mask = ca_mask[..., None] * ca_mask[..., None, :]
    ca_distances = get_pairwise_distances(ca_coords)

    # Identify interface candidates based on distance threshold
    interface_candidates = get_interface_candidates(ca_distances, protein["asym_id"], pair_mask, ca_ca_threshold)

    if torch.any(interface_candidates):
        with numpy_seed(random_seed, key="multimer_spatial_crop"):
            target_res = int(np.random.choice(interface_candidates))
    else:
        return get_contiguous_crop_idx(protein, crop_size, random_seed)

    # Choose a target residue randomly from interface candidates if available
    if torch.any(interface_candidates):
        with numpy_seed(random_seed, key="multimer_spatial_crop"):
            target_res = int(np.random.choice(interface_candidates))
    else:
        # If no interface candidates, revert to contiguous cropping
        return get_contiguous_crop_idx(protein, crop_size, random_seed)

    # Compute distances to the target residue and apply infinity to non-positioned residues
    to_target_distances = ca_distances[target_res]
    to_target_distances[~ca_mask] = inf

    # Apply a small increment to break ties in distances
    break_tie = torch.arange(0, to_target_distances.shape[-1], device=to_target_distances.device).float() * 1e-3
    to_target_distances += break_tie

    # Get the closest residues to the target, sort them, and return
    ret = torch.argsort(to_target_distances)[:crop_size]
    return ret.sort().values


def get_pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    """
    Calculate the pairwise Euclidean distances between a set of coordinates.

    This function computes the Euclidean distance between each pair of points
    represented in the 'coords' tensor. The input tensor is expected to be of shape
    (N, D), where N is the number of points and D is the dimensionality of each point.

    Args:
        coords (torch.Tensor): A tensor of shape (N, D), where each row represents
                               a point in D-dimensional space.

    Returns:
        torch.Tensor: A tensor of shape (N, N) containing the pairwise Euclidean
                      distances between each pair of points in 'coords'.
    """

    # Calculate the difference between each pair of points
    coord_diff = coords.unsqueeze(-2) - coords.unsqueeze(-3)

    # Compute the Euclidean distance for each pair
    return torch.sqrt(torch.sum(coord_diff**2, dim=-1))


def get_interface_candidates(
    ca_distances: torch.Tensor,
    asym_id: torch.Tensor,
    pair_mask: torch.Tensor,
    ca_ca_threshold: float,
) -> torch.Tensor:
    """
    Identify interface candidates based on CA-CA distances and asymmetry IDs.

    This function calculates interface candidates in a molecular structure by
    considering the distances between CA atoms and their asymmetry IDs. Distances
    within the same entity (asymmetry ID) are set to zero. An interface candidate
    is identified if the CA-CA distance is greater than zero and less than the
    specified threshold.

    Args:
        ca_distances (torch.Tensor): A tensor of CA-CA distances.
        asym_id (torch.Tensor): A tensor of asymmetry IDs for each CA atom.
        pair_mask (torch.Tensor): A tensor mask to indicate valid pairs.
        ca_ca_threshold (float): The threshold distance to define an interface.

    Returns:
        torch.Tensor: A tensor containing the indices of interface candidates.

    """
    # Compare asym IDs to determine if atoms are in the same entity
    in_same_asym = asym_id[..., None] == asym_id[..., None, :]

    # Adjust distances by setting those in the same entity to zero and applying the pair mask
    ca_distances = ca_distances * (1.0 - in_same_asym.float()) * pair_mask

    # Count the number of interfaces based on the distance criteria
    cnt_interfaces = torch.sum((ca_distances > 0) & (ca_distances < ca_ca_threshold), dim=-1)

    # Identify the indices of the interface candidates
    interface_candidates = cnt_interfaces.nonzero(as_tuple=True)[0]

    return interface_candidates
