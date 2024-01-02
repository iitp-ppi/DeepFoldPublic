# Copyright 2023 DeepFold Team
# Copyright 2021 DeepMind Technologies Limited


from typing import Mapping, Optional, Sequence

import torch
from omegaconf import DictConfig

from deepfold.common import residue_constants as rc
from deepfold.model.alphafold.data.transforms import NUM_RES, curry1
from deepfold.model.alphafold.data.types import TensorDict
from deepfold.utils.tensor_utils import masked_mean


def gumbel_noise(
    shape: Sequence[int],
    device: torch.device,
    eps: float = 1e-6,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate Gumbel noise of given shape."""

    uniform_noise = torch.rand(shape, generator=generator, dtype=torch.float32, device=device)
    gumbel = -torch.log(-torch.log(uniform_noise + eps) + eps)
    return gumbel


def gumbel_max_sample(logits: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sample from a probability distribution given by logits."""

    z = gumbel_noise(logits.shape, device=logits.device, generator=generator)
    return torch.nn.functional.one_hot(torch.argmax(logits + z, dim=-1), logits.shape[-1])


def gumbel_argsort_sample_idx(logits: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sample with replacement from a distribution given by logits."""

    z = gumbel_noise(logits.shape, device=logits.device, generator=generator)
    return torch.argsort(logits + z, dim=-1, descending=True)


@curry1
def make_masked_msa(
    batch: TensorDict,
    config: DictConfig,
    replace_fraction: float,
    seed: Optional[int] = None,
    eps: float = 1e-6,
) -> TensorDict:
    """Create data from BERT on raw MSA."""

    # Add a random amino acid uniformly
    random_aa = torch.Tensor([0.05] * 20 + [0.0, 0.0], device=batch["msa"].device)

    categorical_probs = (
        config.uniform_prob * random_aa
        + config.profile_prob * batch["msa_profile"]
        + config.same_prob * torch.nn.functional.one_hot(batch["msa"], 22)
    )

    # Put all remaining probability on [MASK] which is a new column.
    mask_prob = 1.0 - config.profile_prob - config.same_prob - config.uniform_prob
    categorical_probs = torch.nn.functional.pad(categorical_probs, [0, 1], value=mask_prob)

    shape = batch["msa"].shape
    mask_position = torch.rand(shape, device=batch["msa"].device) < replace_fraction
    mask_position *= batch["msa_mask"].to(mask_position.dtype)

    logits = torch.log(categorical_probs + eps)

    g = None
    if seed is not None:
        g = torch.Generator(device=batch["msa"].device)
        g.manual_seed(seed)

    bert_msa = gumbel_max_sample(logits, generator=g)
    bert_msa = torch.where(mask_position, torch.argmax(bert_msa, dim=-1), batch["msa"])
    bert_msa *= batch["msa_mask"].to(bert_msa.dtype)

    # Mix real and masked MSA
    if "bert_mask" in batch:
        batch["bert_mask"] *= mask_position.to(torch.float32)
    else:
        batch["bert_mask"] = mask_position.to(torch.float32)
    batch["true_msa"] = batch["msa"]
    batch["msa"] = bert_msa

    return batch


@curry1
def nearest_neighbor_clusters(batch: TensorDict, gap_agreement_weight: float = 0.0) -> TensorDict:
    """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

    device = batch["msa_mask"].device

    # Determine how much weight we assign to each agreement.
    # In theory, we could use a full BLOSUM matrix here,
    # but right now let's just down-weight gap agreement
    # because it could be spurious.
    # Never put weight on agreeing on BERT mask.

    weights = torch.Tensor([1.0] * 21 + [gap_agreement_weight] + [0.0], device=device)

    msa_mask = batch["msa_mask"]
    msa_one_hot = torch.nn.functional.one_hot(batch["msa"], 23)

    extra_mask = batch["extra_msa_mask"]
    extra_one_hot = torch.nn.functional.one_hot(batch["extra_msa"], 23)

    msa_one_hot_masked = msa_mask[:, :, None] * msa_one_hot
    extra_one_hot_masked = extra_mask[:, :, None] * extra_one_hot

    agreement = torch.einsum("mrc,nrc->nm", extra_one_hot_masked, weights * msa_one_hot_masked)

    cluster_assignment = torch.nn.functional.softmax(1e3 * agreement, dim=0)
    cluster_assignment *= torch.einsum("mr,nr->mn", msa_mask, extra_mask)

    cluster_count = torch.sum(cluster_assignment, dim=-1)
    cluster_count += 1.0  # Always include the sequence itself

    msa_sum = torch.einsum("nm,mrc->nrc", cluster_assignment, extra_one_hot_masked)
    msa_sum += msa_one_hot_masked

    cluster_profile = msa_sum / cluster_count[:, None, None]

    extra_deletion_matrix = batch["extra_deletion_matrix"]
    deletion_matrix = batch["deletion_matrix"]

    del_sum = torch.einsum("nm,mc->nc", cluster_assignment, extra_mask * extra_deletion_matrix)
    del_sum += deletion_matrix  # Origional sequence
    cluster_deletion_mean = del_sum / cluster_count[:, None]

    batch["cluster_profile"] = cluster_profile
    batch["cluster_deletion_mean"] = cluster_deletion_mean

    return batch


def create_target_feat(batch: TensorDict) -> TensorDict:
    """Create the target features."""

    batch["target_feat"] = torch.nn.functional.one_hot(batch["aatype"], 21).to(torch.float32)
    return batch


def create_msa_feat(batch: TensorDict) -> TensorDict:
    """Create and concatenate MSA features."""

    msa_1hot = torch.nn.functional.one_hot(batch["msa"], 23)
    deletion_matrix = batch["deletion_matrix"]
    has_deletion = torch.clamp(deletion_matrix, min=0.0, max=1.0)[..., None]
    deletion_value = (torch.atan(deletion_matrix / 3.0) * (2.0 / torch.pi))[..., None]
    deletion_mean_value = (torch.atan(batch["cluster_deletion_mean"] / 3.0) * (2.0 / torch.pi))[..., None]

    msa_feat = torch.cat(
        [
            msa_1hot,
            has_deletion,
            deletion_value,
            batch["cluster_profile"],
            deletion_mean_value,
        ],
        dim=-1,
    )

    batch["msa_feat"] = msa_feat

    return batch


@curry1
def create_extra_msa_feat(batch: TensorDict, num_extra_msa: int) -> TensorDict:
    """
    Expand extra_msa into one-hot and concat with other extra msa features.

    Notes:
        Do this as late as possible as the one hot extra msa can be very large.

    Args:
        batch: TensorDict
            a dictionary with the following keys:
            - `extra_msa`: [num_seq, num_res]
                MSA that wasn't selected as a cluster center.
                This isn't one-hotted.
            - `extra_deletion_matrix`: [num_seq, num_res]
                Number of deletions at given position.
        num_extra_msa: int
            number of extra msa to use.
    Returns:
        concatenated tensor of extra MSA features.
    """

    # 23 = 20 amino acids + 'X' for unknown + gap + BERT mask
    extra_msa = batch["extra_msa"][:num_extra_msa]
    deletion_matrix = batch["extra_deletion_matrix"][:num_extra_msa]
    has_deletion = torch.clip(deletion_matrix, 0.0, 1.0)
    deletion_value = torch.atan(deletion_matrix / 3.0) * (2.0 / torch.pi)
    extra_msa_mask = batch["extra_msa_mask"][:num_extra_msa]

    batch["extra_msa"] = extra_msa
    batch["extra_msa_mask"] = extra_msa_mask
    batch["extra_msa_has_deletion"] = has_deletion
    batch["extra_msa_deletion_value"] = deletion_value

    return batch


@curry1
def sample_msa(
    batch: TensorDict,
    max_seq: int,
    max_extra_msa_seq: int,
    seed: Optional[int] = None,
    inf: float = 1e6,
) -> TensorDict:
    """Sample MSA randomly, remaining sequences are stored as `extra_*`."""

    g = None
    if seed is not None:
        g = torch.Generator(device=batch["msa"].device)
        g.manual_seed(seed)

    # Sample uniformly among sequences with at least one non-masked position
    logits = (torch.clamp(torch.sum(batch["msa_mask"], dim=-1), 0.0, 1.0) - 1.0) * inf
    # The cluster_bias_mask can be used to preserve the first row for each chain
    if "cluster_bias_mask" not in batch:
        cluster_bias_mask = torch.nn.functional.pad(
            batch["msa"].new_zeros(batch["msa"].shape[0] - 1),
            (1, 0),
            value=1.0,
        )
    else:
        cluster_bias_mask = batch["cluster_bias_mask"]

    logits += cluster_bias_mask * inf
    index_order = gumbel_argsort_sample_idx(logits, generator=g)
    sel_idx = index_order[:max_seq]
    extra_idx = index_order[max_seq:][:max_extra_msa_seq]

    for key in ("msa", "deletion_matrix", "msa_mask", "bert_mask"):
        if key in batch:
            batch[f"extra_{key}"] = batch[key][extra_idx]
            batch[key] = batch[key][sel_idx]

    return batch


def make_msa_profile(batch):
    """Compute the MSA profile."""

    # Compute the profile for every residue (over all MSA sequences).
    batch["msa_profile"] = masked_mean(
        batch["msa_mask"][..., None],
        torch.nn.functional.one_hot(batch["msa"], 22),
        dim=-3,
    )

    return batch


def randint(lower: int, upper: int, generator: torch.Generator, device: torch.device) -> int:
    """Sample random integer from [lower, upper]."""
    return int(torch.randint(lower, upper + 1, (1,), device=device, generator=generator).item())


def get_interface_residues(
    positions: torch.Tensor,
    atom_mask: torch.Tensor,
    asym_id: torch.Tensor,
    interface_threshold: float,  # Angstrom
) -> torch.Tensor:
    coord_diff = positions[..., :, None, :, :] - positions[..., None, :, :, :]
    pairwise_dists = torch.sqrt(torch.sum(coord_diff**2, dim=-1))

    diff_chain_mask = (asym_id[..., None, :] != asym_id[..., :, None]).float()
    pair_mask = atom_mask[..., :, None, :] * atom_mask[..., None, :, :]
    mask = (diff_chain_mask[..., None] * pair_mask).bool()

    min_dist_per_res, _ = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1)

    valid_interfaces = torch.sum((min_dist_per_res < interface_threshold).float(), dim=-1)
    interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]

    return interface_residues_idxs


def get_spatial_crop_idx(
    batch: TensorDict,
    crop_size: int,
    interface_threshold: float,
    generator: torch.Generator,
):
    positions = batch["all_atom_positions"]
    atom_mask = batch["all_atom_mask"]
    asym_id = batch["asym_id"]

    interface_residues = get_interface_residues(
        positions=positions, atom_mask=atom_mask, asym_id=asym_id, interface_threshold=interface_threshold
    )

    if not torch.any(interface_residues):
        return get_contiguous_crop_idx(batch, crop_size, generator)

    target_res_idx = randint(
        lower=0, upper=interface_residues.shape[-1] - 1, generator=generator, device=positions.device
    )

    target_res = interface_residues[target_res_idx]

    ca_idx = rc.atom_order["CA"]
    ca_positions = positions[..., ca_idx, :]
    ca_mask = atom_mask[..., ca_idx].bool()

    coord_diff = ca_positions[..., None, :] - ca_positions[..., None, :, :]
    ca_pairwise_dists = torch.sqrt(torch.sum(coord_diff**2, dim=-1))

    to_target_distances = ca_pairwise_dists[target_res]
    break_tie = torch.arange(0, to_target_distances.shape[-1], device=positions.device).float() * 1e-3
    to_target_distances = torch.where(ca_mask, to_target_distances, torch.inf) + break_tie

    ret = torch.argsort(to_target_distances)[:crop_size]
    return ret.sort().values


def get_contiguous_crop_idx(
    batch: TensorDict,
    crop_size: int,
    generator: torch.Generator,
):
    unique_asym_ids, chain_idxs, chain_lens = batch["asym_id"].unique(dim=-1, return_inverse=True, return_counts=True)

    shuffle_idx = torch.randperm(chain_lens.shape[-1], device=chain_lens.device, generator=generator)

    _, idx_sorted = torch.sort(chain_idxs, stable=True)
    cum_sum = chain_lens.cumsum(dim=0)
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]), dim=0)
    asym_offsets = idx_sorted[cum_sum]

    num_budget = crop_size
    num_remaining = int(batch["seq_length"])

    crop_idxs = []
    for idx in shuffle_idx:
        chain_len = int(chain_lens[idx])
        num_remaining -= chain_len

        crop_size_max = min(num_budget, chain_len)
        crop_size_min = min(chain_len, max(0, num_budget - num_remaining))
        chain_crop_size = randint(
            lower=crop_size_min,
            upper=crop_size_max,
            generator=generator,
            device=chain_lens.device,
        )

        num_budget -= chain_crop_size

        chain_start = randint(lower=0, upper=chain_len - chain_crop_size, generator=generator, device=chain_lens.device)

        asym_offset = asym_offsets[idx]
        crop_idxs.append(torch.arange(asym_offset + chain_start, asym_offset + chain_start + chain_crop_size))

    return torch.concat(crop_idxs).sort().values


@curry1
def random_crop_to_size(
    batch: TensorDict,
    crop_size: int,
    max_templates: int,
    shape_schema: Mapping[str, Sequence[Optional[str]]],
    spatial_crop_prob: float,
    interface_threshold: float,
    subsample_templates: bool = False,
    seed: Optional[int] = None,
) -> TensorDict:
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""

    # We want each ensemble to be cropped the same way
    g = None
    if seed is not None:
        g = torch.Generator(device=batch["seq_length"].device)
        g.manual_seed(seed)

    use_spatial_crop = torch.rand((1,), device=batch["seq_length"].device, generator=g) < spatial_crop_prob

    num_res = batch["aatype"].shape[0]
    if num_res <= crop_size:
        crop_idxs = torch.arange(num_res)
    elif use_spatial_crop:
        crop_idxs = get_spatial_crop_idx(batch, crop_size, interface_threshold, g)
    else:
        crop_idxs = get_contiguous_crop_idx(batch, crop_size, g)

    if "template_mask" in batch:
        num_templates = batch["template_mask"].shape[-1]
    else:
        num_templates = 0

    # Don't need to subsample templates if there is no template.
    subsample_templates = subsample_templates and num_templates

    if subsample_templates:
        templates_crop_start = randint(lower=0, upper=num_templates, generator=g, device=batch["seq_length"].device)
        templates_select_indices = torch.randperm(num_templates, device=batch["seq_length"].device, generator=g)
    else:
        templates_crop_start = 0

    num_res_crop_size = min(int(batch["seq_length"]), crop_size)
    num_templates_crop_size = min(num_templates - templates_crop_start, max_templates)

    for k, v in batch.items():
        if k not in shape_schema or ("template" not in k and NUM_RES not in shape_schema[k]):
            continue

        # Randomly permute the templates before cropping them
        if k.startswith("template") and subsample_templates:
            v = v[templates_select_indices]

        for i, (dim_size, dim) in enumerate(zip(shape_schema[k], v.shape)):
            is_num_res = dim_size == NUM_RES
            if i == 0 and k.startswith("template"):
                v = v[slice(templates_crop_start, templates_crop_start + num_templates_crop_size)]
            elif is_num_res:
                v = torch.index_select(v, i, crop_idxs)

        batch[k] = v

    batch["seq_length"] = batch["seq_length"].new_tensor(num_res_crop_size)

    return batch
