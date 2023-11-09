# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited


import itertools
from functools import reduce, wraps
from operator import add
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig

import deepfold.common.residue_constants as rc
from deepfold.model.alphafold.pipeline.types import TensorDict
from deepfold.utils.geometry import Rigid, Rotation
from deepfold.utils.tensor_utils import batched_gather


class TransformFn(Protocol):
    def __call__(self, p: TensorDict, **kwargs: Any) -> TensorDict:
        ...


MSA_FEATURE_NAMES = [
    "msa",
    "deletion_matrix",
    "msa_mask",
    "msa_row_mask",
    "bert_mask",
    "true_msa",
]


def _make_one_hot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)

    return x_one_hot


def map_fn(fn: TransformFn, x: TensorDict) -> TensorDict:
    ensembles = [fn(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = torch.stack([dict_i[feat] for dict_i in ensembles], dim=-1)

    return ensembled_dict


def curry1(f: TransformFn) -> TensorDict:
    @wraps(f)
    def func(*args, **kwargs):
        return lambda p: f(p, *args, **kwargs)

    return func


@curry1
def compose(x: TensorDict, fs: Sequence[TransformFn]):
    for f in fs:
        x = f(x)

    return x


# @curry1
def cast_to_int64(p: TensorDict) -> TensorDict:
    for k, v in p.items():
        if v.dtype == torch.int32:
            p[k] = v.type(torch.int64)

    return p


# @curry1
def make_seq_mask(p: TensorDict) -> TensorDict:
    p["seq_mask"] = torch.ones(p["aatype"].shape, dtype=torch.float32)

    return p


# @curry1
def make_template_mask(p: TensorDict) -> TensorDict:
    p["template_mask"] = torch.ones(p["template_aatype"].shape[0], dtype=torch.float32)

    return p


# @curry1
def make_all_atom_aatype(p: TensorDict) -> TensorDict:
    p["all_atom_aatype"] = p["aatype"]

    return p


# @curry1
def fix_templates_aatype(p: TensorDict) -> TensorDict:
    """Convert hhsearch-aatype one-hot to ours."""
    # Map one-hot to indices
    num_templates = p["template_aatype"].shape[0]
    if num_templates > 0:
        p["template_aatype"] = torch.argmax(p["template_aatype"], dim=-1)
        # Convert hhsearch-aatype to ours
        new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
        new_order = torch.tensor(new_order_list, dtype=torch.int64, device=p["aatype"].device).expand(num_templates, -1)
        p["template_aatype"] = torch.gather(new_order, 1, index=p["template_aatype"])

    return p


# @curry1
def correct_msa_restypes(p: TensorDict) -> TensorDict:
    """Correct MSA restype."""
    new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = torch.tensor([new_order_list] * p["msa"].shape[1], device=p["msa"].device).transpose(0, 1)
    p["msa"] = torch.gather(new_order, 0, p["msa"])

    perm_matrix = np.zeros((22, 22), dtype=np.float32)
    perm_matrix[range(len(new_order_list)), new_order_list] = 1.0

    for k in p:
        if "profile" in k:
            num_dim = p[k].shape.as_list()[-1]
            assert num_dim in (20, 21, 22), f"num_dim for {k} out of expected range: {num_dim}"
            p[k] = torch.dot(p[k], perm_matrix[:num_dim, :num_dim])

    return p


# @curry1
def squeeze_features(p: TensorDict) -> TensorDict:
    """Remove sigleton and repeated dimensions in features."""
    p["aatype"] = torch.argmax(p["aatype"], dim=-1)
    for k in [
        "domain_name",
        "msa",
        "num_alignments",
        "seq_length",
        "sequence",
        "superfamily",
        "deletion_matrix",
        "resolution",
        "between_segment_residues",
        "residue_index",
        "template_all_atom_mask",
    ]:
        if k in p:
            final_dim = p[k].shape[-1]
            if isinstance(final_dim, int) and final_dim == 1:
                if torch.is_tensor(p[k]):
                    p[k] = torch.squeeze(p[k], dim=-1)
                else:
                    p[k] = np.squeeze(p[k], axis=-1)

    for k in ["seq_length", "num_alignments"]:
        if k in p:
            p[k] = p[k][0]

    return p


@curry1
def randomly_replace_msa_with_unknown(p: TensorDict, replace_proportion: float) -> TensorDict:
    """Replace a portion of the MSA with 'X'."""
    msa_mask = torch.rand(p["msa"].shape) < replace_proportion
    x_idx = 20
    gap_idx = 21
    msa_mask = torch.logical_and(msa_mask, p["msa"] != gap_idx)
    p["msa"] = torch.where(msa_mask, torch.ones_like(p["msa"]) * x_idx, p["msa"])
    aatype_mask = torch.rand(p["aatype"].shape) < replace_proportion
    p["aatype"] = torch.where(aatype_mask, torch.ones_like(p["aatype"]) * x_idx, p["aatype"])

    return p


@curry1
def sample_msa(p: TensorDict, max_seq: int, keep_extra: bool, seed: Optional[int] = None) -> TensorDict:
    """Sample MSA randomly, remaining sequences are stored as extra."""
    num_seq = p["msa"].shape[0]

    g = None
    if seed is not None:
        g = torch.Generator(device=p["msa"].device)
        g.manual_seed(seed)

    shuffled = torch.randperm(num_seq - 1, generator=g) + 1
    index_order = torch.cat((torch.tensor([0], device=shuffled.device), shuffled), dim=0)
    num_sel = min(max_seq, num_seq)
    sel_seq, not_sel_seq = torch.split(index_order, [num_sel, num_seq - num_sel])

    for k in MSA_FEATURE_NAMES:
        if k in p:
            if keep_extra:
                p[f"extra_{k}"] = torch.index_select(p[k], 0, not_sel_seq)
            p[k] = torch.index_select(p[k], 0, sel_seq)

    return p


@curry1
def add_distillation_flag(p: TensorDict, distillation: torch.Tensor) -> TensorDict:
    p["is_distillation"] = distillation

    return p


@curry1
def sample_msa_distillation(p: TensorDict, max_seq: int) -> TensorDict:
    if p["is_distillation"] == 1:
        p = sample_msa(max_seq, keep_extra=False)(p)

    return p


@curry1
def crop_extra_msa(p: TensorDict, max_extra_msa: int) -> TensorDict:
    num_seq = p["extra_msa"].shape[0]
    num_sel = min(max_extra_msa, num_seq)
    select_indices = torch.randperm(num_seq)[:num_sel]
    for k in MSA_FEATURE_NAMES:
        if f"extra_{k}" in p:
            p[f"extra_{k}"] = torch.index_select(p[f"extra_{k}"], 0, select_indices)

    return p


# @curry1
def delete_extra_msa(p: TensorDict) -> TensorDict:
    for k in MSA_FEATURE_NAMES:
        if f"extra_{k}" in p:
            del p[f"extra_{k}"]

    return p


@curry1
def nearest_neighbor_clusters(p: TensorDict, gap_agreement_weight: float = 0.0) -> TensorDict:
    weights = torch.cat(
        [
            torch.ones(21, device=p["msa"].device),
            gap_agreement_weight * torch.ones(1, device=p["msa"].device),
            torch.zeros(1, device=p["msa"].device),
        ],
        dim=0,
    )

    # Make agreement score as weighted Hamming distance
    msa_one_hot = _make_one_hot(p["msa"], 23)
    sample_one_hot = p["msa_mask"][:, :, None] * msa_one_hot
    extra_msa_one_hot = _make_one_hot(p["extra_msa"], 23)
    extra_one_hot = p["extra_msa_mask"][:, :, None] * extra_msa_one_hot

    num_seq, num_res, _ = sample_one_hot.shape
    extra_num_seq, _, _ = extra_one_hot.shape

    # Compute einsum("mrc,nrc,c->mn", sample_one_hot, extra_one_hot, weights)
    agreement = torch.matmul(
        torch.reshape(extra_one_hot, [extra_num_seq, num_res * 23]),
        torch.reshape(sample_one_hot * weights, [num_seq, num_res * 23]).transpose(0, 1),
    )

    # Assign each sequence in the extra sequences to the closest MSA sample
    p["extra_cluster_assignment"] = torch.argmax(agreement, dim=1).to(torch.int64)

    return p


def _unsorted_segment_sum(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """
    Compute the sum along segments of a tensor.

    Args:
        data: torch.Tensor
            A tensor whose segments to be summed.
        segment_ids: torch.Tensor
            The 1-dimensional segment indice tensor.
        num_segments: torch.Tensor
            The number of segments.

    Returns:
        A tensor of some data type as the data argument.
    """

    assert len(segment_ids.shape) == 1 and segment_ids.shape[0] == data.shape[0]
    segment_ids = segment_ids.view(segment_ids.shape[0], *((1,) * len(data.shape[1:])))
    segment_ids = segment_ids.expand(data.shape)
    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape, device=segment_ids.device).scatter_add_(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)

    return tensor


@curry1
def summarize_clusters(p: TensorDict) -> TensorDict:
    """Produce profile and deletion_matrix_mean within each cluster."""

    num_seq = p["msa"].shape[0]

    def csum(x):
        return _unsorted_segment_sum(x, p["extra_cluster_assignment"], num_seq)

    mask = p["extra_msa_mask"]
    mask_counts = 1e-6 + p["msa_mask"] + csum(mask)  # Include center

    msa_sum = csum(mask[:, :, None] * _make_one_hot(p["extra_msa"], 23))
    msa_sum += _make_one_hot(p["msa"], 23)  # Original sequence
    p["cluster_profile"] = msa_sum / mask_counts[:, :, None]
    del msa_sum

    del_sum = csum(mask * p["extra_deletion_matrix"])
    del_sum += p["deletion_matrix"]  # Original sequence
    p["cluster_deletion_mean"] = del_sum / mask_counts
    del del_sum

    return p


# @curry1
def make_msa_mask(p: TensorDict) -> TensorDict:
    p["msa_mask"] = torch.ones(p["msa"].shape, dtype=torch.float32)
    p["msa_row_mask"] = torch.ones(p["msa"].shape[0], dtype=torch.float32)

    return p


def _pseudo_beta_helper(
    aatype: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create pseudo-beta features."""

    # C-alpha for GLY
    is_gly = torch.eq(aatype, rc.restype_order["G"])
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]

    pseudo_beta = torch.where(
        torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = torch.where(is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx])
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


@curry1
def make_pseudo_beta(p: TensorDict, prefix="") -> TensorDict:
    """Create pseudo-beta position and mask."""
    assert prefix in ["", "template_"]

    pos, mask = _pseudo_beta_helper(
        p["template_aatype" if prefix else "aatype"],
        p[f"{prefix}all_atom_positions"],
        p["template_all_atom_mask" if prefix else "all_atom_mask"],
    )
    p[f"{prefix}pseudo_beta"], p[f"{prefix}pseudo_beta_mask"] = pos, mask

    return p


@curry1
def add_constant_field(p: TensorDict, key: str, value: Any) -> TensorDict:
    p[key] = torch.tensor(value, device=p["msa"].device)

    return p


def _shaped_categorical(probs: torch.Tensor, eps=1e-10) -> torch.Tensor:
    ds = probs.shape
    num_classes = ds[-1]
    distribution = torch.distributions.categorical.Categorical(torch.reshape(probs + eps, [-1, num_classes]))
    counts = distribution.sample()

    return torch.reshape(counts, ds[:-1])


# @curry1
def make_hhblits_profile(p: TensorDict) -> TensorDict:
    """Compute the HHblits MSA profile if not already present."""

    if "hhblilts_profile" in p:
        return p

    # Compute the profile for every residue (over all MSA seqs).
    msa_one_hot = _make_one_hot(p["msa"], 22)

    p["hhblits_profile"] = torch.mean(msa_one_hot, dim=0)

    return p


@curry1
def make_masked_msa(p: TensorDict, masked_msa_cfg: DictConfig, replace_fraction: float) -> TensorDict:
    """Create data for BERT on raw MSA."""

    # Add a random amino acid uniformly; 20 + 1 + 1 = 22
    random_aa = torch.tensor([0.05] * 20 + [0.0, 0.0], dtype=torch.float32, device=p["aatype"].device)

    categorical_probs = (
        masked_msa_cfg.uniform_prob * random_aa
        + masked_msa_cfg.profile_prob * p["hhblits_profile"]
        + masked_msa_cfg.same_prob * _make_one_hot(p["msa"], 22)
    )

    # Put all remaining probability on [MASK] which is a new column
    pad_shapes = list(reduce(add, [(0, 0) for _ in range(len(categorical_probs.shape))]))
    pad_shapes[1] = 1
    mask_prob = 1.0 - masked_msa_cfg.uniform_prob - masked_msa_cfg.profile_prob - masked_msa_cfg.same_prob
    assert mask_prob >= 0.0

    categorical_probs = torch.nn.functional.pad(categorical_probs, pad_shapes, value=mask_prob)

    sh = p["msa"].shape
    mask_position = torch.rand(sh) < replace_fraction

    bert_msa = _shaped_categorical(categorical_probs)
    bert_msa = torch.where(mask_position, bert_msa, p["msa"])

    # Mix real and masked MSA
    p["bert_mask"] = mask_position.to(torch.float32)
    p["true_msa"] = p["msa"]
    p["msa"] = bert_msa

    return p


@curry1
def make_fixed_size(
    p: TensorDict,
    shape_schema: Mapping[str, Sequence[Optional[str]]],  # From cfg
    msa_cluster_size: int,
    extra_msa_size: int,
    num_res: int = 0,
    num_templates=0,
) -> TensorDict:
    """
    Make MSA and sequence dimension to be fixed size.

    Args:
        p: TensorDict
            Dictionary of feature tensors.
        shape_schema: Mapping[str, Sequence[Optional[str]]]
            Mapping from a feature key to feature shape.
        ...

    Returns:
        Transformed TensorDict.
    """

    pad_size_map = {
        "NUM_RES": num_res,
        "NUM_MSXA_SEQ": msa_cluster_size,
        "NUM_EXTRA_SEQ": extra_msa_size,
        "NUM_TEMPLATES": num_templates,
    }

    excludes = [
        "extra_cluster_assignment",
    ]
    for k, v in p.items():
        # Don't transfer this to the accelerator
        if k in excludes:
            continue
        shape = list(v.shape)
        schema = shape_schema[k]
        msg = "Rank mismatch between shape and schema for"
        assert len(shape) == len(schema), f"{msg} {k}: {shape} != {schema}"
        pad_size = [pad_size_map.get(s2, None) or s1 for s1, s2 in zip(shape, schema)]

        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        padding.reverse()
        padding = list(itertools.chain(*padding))
        if padding:
            p[k] = torch.nn.functional.pad(v, padding)
            p[k] = torch.reshape(p[k], pad_size)

    return p


@curry1
def make_msa_feat(p: TensorDict) -> TensorDict:
    """Create and concatenate MSA features."""

    # Always zero for chain, but keeping for compatibility
    has_break = torch.clip(p["between_segment_residues"].to(torch.float32), 0.0, 1.0)
    aatype_one_hot = _make_one_hot(p["aatype"], 21)

    target_feat = [torch.unsqueeze(has_break, dim=-1), aatype_one_hot]

    msa_one_hot = _make_one_hot(p["msa"], 23)
    has_deletion = torch.clip(p["deletion_matrix"], 0.0, 1.0)
    deletion_value = torch.atan(p["deletion_matrix"] / 3.0) * (2.0 / np.pi)

    msa_feat = [
        msa_one_hot,
        torch.unsqueeze(has_deletion, dim=-1),
        torch.unsqueeze(deletion_value, dim=-1),
    ]

    if "cluster_profile" in p:
        deletion_mean_value = torch.atan(p["cluster_deletion_mean"] / 3.0) * (2.0 / np.pi)
        msa_feat.extend([p["cluster_profile"], torch.unsqueeze(deletion_mean_value, dim=-1)])

    if "extra_deletion_matrix" in p:
        p["extra_has_deletion"] = torch.clip(p["extra_deletion_matrix"], 0.0, 1.0)
        p["extra_deletion_value"] = torch.atan(p["extra_deletion_matrix"] / 3.0) * (2.0 / np.pi)

    p["msa_feat"] = torch.cat(msa_feat, dim=-1)
    p["target_feat"] = torch.cat(target_feat, dim=-1)

    return p


@curry1
def select_features(p: TensorDict, feature_list: Sequence[str]) -> TensorDict:
    return {k: v for k, v in p.items() if k in feature_list}


@curry1
def crop_templates(p: TensorDict, max_templates: int):
    for k, v in p.items():
        if k.startswith("template_"):
            p[k] = v[:max_templates]

    return p


# @curry1
def make_atom14_masks(p: TensorDict) -> TensorDict:
    """Construct tensor atom positions (14 instead of 37)."""

    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append([(rc.atom_order[name] if name else 0) for name in atom_names])
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0 for name in rc.atom_types]
        )
        restype_atom14_mask.append([1.0 if name else 0.0 for name in atom_names])

    # Add dummy mapping for 'X'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = torch.tensor(restype_atom14_to_atom37, dtype=torch.int32, device=p["aatype"].device)
    restype_atom37_to_atom14 = torch.tensor(restype_atom37_to_atom14, dtype=torch.int32, device=p["aatype"].device)
    restype_atom14_mask = torch.tensor(restype_atom14_mask, dtype=torch.float32, device=p["aatype"].device)
    protein_aatype = p["aatype"].to(torch.long)

    # Create the mapping for (residx, atom14) -> atom37,
    # i.e., an array with shape (N, 14) containing the atom37 indices
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    p["atom14_atom_exists"] = residx_atom14_mask
    p["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

    # Create the gather indices for mapping back
    residx_atom37_to_atom14 = residx_atom14_mask
    p["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

    # Create the corresponding mask
    restype_atom37_mask = torch.zeros([21, 37], dtype=torch.float32, device=p["aatype"].device)
    for restype, res_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[res_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    p["atom37_atom_exists"] = residx_atom37_mask

    return p


# @curry1
def make_atom14_positions(p: TensorDict) -> TensorDict:
    """Constructs denser atom positions (14 dimensions instead of 37)."""

    residx_atom14_mask = p["atom14_atom_exists"]
    residx_atom14_to_atom37 = p["residx_atom14_to_atom37"]

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * batched_gather(
        p["all_atom_mask"], residx_atom14_to_atom37, dim=-1, no_batch_dims=len(p["all_atom_mask"].shape[:-1])
    )

    # Gather the ground truth positions.
    residx_atom14_gt_positions = residx_atom14_gt_mask[..., None] * (
        batched_gather(
            p["all_atom_positions"],
            residx_atom14_to_atom37,
            dim=-2,
            no_batch_dims=len(p["all_atom_positions"].shape[:-2]),
        )
    )

    p["atom14_atom_exists"] = residx_atom14_mask
    p["atom14_gt_exists"] = residx_atom14_gt_mask
    p["atom14_gt_positions"] = residx_atom14_gt_positions

    # As the atom naming is ambiguous for 7 of the 20 amino acids,
    # provide alternative ground truth coordinates where the naming is swapped
    restype_3 = [rc.restype_1to3[res] for res in rc.restypes]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {
        res: torch.eye(14, dtype=p["all_atom_mask"].dtype, device=p["all_atom_mask"].device) for res in restype_3
    }
    for resname, swap in rc.residue_atom_renaming_swaps.items():
        correspondences = torch.arange(14, device=p["all_atom_mask"].device)
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = rc.restype_name_to_atom14_names[resname].index(source_atom_swap)
            target_index = rc.restype_name_to_atom14_names[resname].index(target_atom_swap)
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = p["all_atom_mask"].new_zeros((14, 14))
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.0
        all_matrices[resname] = renaming_matrix

    renaming_matrices = torch.stack([all_matrices[restype] for restype in restype_3])

    # Pick the transformation matrices for the given residue sequence shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[p["aatype"]]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = torch.einsum("...rac,...rab->...rbc", residx_atom14_gt_positions, renaming_transform)
    p["atom14_alt_gt_positions"] = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the ground truth mask,
    # if only one of the atoms in an ambiguous pair has a ground truth position).
    alternative_gt_mask = torch.einsum("...ra,...rab->...rb", residx_atom14_gt_mask, renaming_transform)
    p["atom14_alt_gt_exists"] = alternative_gt_mask

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = p["all_atom_mask"].new_zeros((21, 14))
    for resname, swap in rc.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = rc.restype_order[rc.restype_3to1[resname]]
            atom_idx1 = rc.restype_name_to_atom14_names[resname].index(atom_name1)
            atom_idx2 = rc.restype_name_to_atom14_names[resname].index(atom_name2)
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    p["atom14_atom_is_ambiguous"] = restype_atom14_is_ambiguous[p["aatype"]]

    return p


@curry1
def atom37_to_frames(p: TensorDict, eps=1e-8) -> TensorDict:
    aatype = p["aatype"]
    all_atom_positions = p["all_atom_positions"]
    all_atom_mask = p["all_atom_mask"]

    batch_dims = len(aatype.shape[:-1])

    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
    restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]

    for restype, restype_letter in enumerate(rc.restypes):
        resname = rc.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if rc.chi_angles_mask[restype][chi_idx]:
                names = rc.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[restype, chi_idx + 4, :] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*aatype.shape[:-1], 21, 8),
    )
    restype_rigidgroup_mask[..., 0] = 1
    restype_rigidgroup_mask[..., 3] = 1
    restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(rc.chi_angles_mask)

    lookuptable = rc.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom37_idx = lookup(restype_rigidgroup_base_atom_names)
    restype_rigidgroup_base_atom37_idx = aatype.new_tensor(restype_rigidgroup_base_atom37_idx)
    restype_rigidgroup_base_atom37_idx = restype_rigidgroup_base_atom37_idx.view(
        *((1,) * batch_dims), *restype_rigidgroup_base_atom37_idx.shape
    )

    residx_rigidgroup_base_atom37_idx = batched_gather(
        restype_rigidgroup_base_atom37_idx, aatype, dim=-3, no_batch_dims=batch_dims
    )

    base_atom_pos = batched_gather(
        all_atom_positions, residx_rigidgroup_base_atom37_idx, dim=-2, no_batch_dims=len(all_atom_positions.shape[:-2])
    )

    gt_frames = Rigid.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=eps,
    )

    group_exists = batched_gather(restype_rigidgroup_mask, aatype, dim=-2, no_batch_dims=batch_dims)

    gt_atoms_exist = batched_gather(
        all_atom_mask, residx_rigidgroup_base_atom37_idx, dim=-1, no_batch_dims=len(all_atom_mask.shape[:-1])
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 8, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(rot_mats=rots)

    gt_frames = gt_frames.compose(Rigid(rots, None))

    restype_rigidgroup_is_ambiguous = all_atom_mask.new_zeros(*((1,) * batch_dims), 21, 8)
    restype_rigidgroup_rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    restype_rigidgroup_rots = torch.tile(restype_rigidgroup_rots, (*((1,) * batch_dims), 21, 8, 1, 1))

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
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = batched_gather(
        restype_rigidgroup_rots,
        aatype,
        dim=-4,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = Rotation(rot_mats=residx_rigidgroup_ambiguity_rot)
    alt_gt_frames = gt_frames.compose(Rigid(residx_rigidgroup_ambiguity_rot, None))

    gt_frames_tensor = gt_frames.to_tensor_4x4()
    alt_gt_frames_tensor = alt_gt_frames.to_tensor_4x4()

    p["rigidgroups_gt_frames"] = gt_frames_tensor
    p["rigidgroups_gt_exists"] = gt_exists
    p["rigidgroups_group_exists"] = group_exists
    p["rigidgroups_group_is_ambiguous"] = residx_rigidgroup_is_ambiguous
    p["rigidgroups_alt_gt_frames"] = alt_gt_frames_tensor

    return p


def _get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.restypes:
        residue_name = rc.restype_1to3[residue_name]
        residue_chi_angles = rc.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([rc.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return chi_atom_indices


@curry1
def atom37_to_torsion_angles(p: TensorDict, prefix="") -> TensorDict:
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 37, 3] atom positions (in atom37
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 37] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """

    aatype = p[prefix + "aatype"]
    all_atom_positions = p[prefix + "all_atom_positions"]
    all_atom_mask = p[prefix + "all_atom_mask"]

    aatype = torch.clamp(aatype, max=20)

    pad = all_atom_positions.new_zeros([*all_atom_positions.shape[:-3], 1, 37, 3])
    prev_all_atom_positions = torch.cat([pad, all_atom_positions[..., :-1, :, :]], dim=-3)

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat([prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]], dim=-2)
    phi_atom_pos = torch.cat([prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]], dim=-2)
    psi_atom_pos = torch.cat([all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]], dim=-2)

    pre_omega_mask = torch.prod(prev_all_atom_mask[..., 1:3], dim=-1) * torch.prod(all_atom_mask[..., :2], dim=-1)
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
    psi_mask = torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype) * all_atom_mask[..., 4]

    chi_atom_indices = torch.as_tensor(_get_chi_atom_indices(), device=aatype.device)

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

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [pre_omega_mask[..., None], phi_mask[..., None], psi_mask[..., None], chis_mask], dim=-1
    )

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :], torsions_atom_pos[..., 2, :], torsions_atom_pos[..., 0, :], eps=1e-8
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(torsions_atom_pos[..., 3, :])

    torsion_angles_sin_cos = torch.stack([fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1)

    denom = torch.sqrt(
        torch.sum(torch.square(torsion_angles_sin_cos), dim=-1, dtype=torsion_angles_sin_cos.dtype, keepdims=True)
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = (
        torsion_angles_sin_cos
        * all_atom_mask.new_tensor(
            [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
    )

    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(rc.chi_pi_periodic)[aatype, ...]

    mirror_torsion_angles = torch.cat([all_atom_mask.new_ones(*aatype.shape, 3), 1.0 - 2.0 * chi_is_ambiguous], dim=-1)

    alt_torsion_angles_sin_cos = torsion_angles_sin_cos * mirror_torsion_angles[..., None]

    p[prefix + "torsion_angles_sin_cos"] = torsion_angles_sin_cos
    p[prefix + "alt_torsion_angles_sin_cos"] = alt_torsion_angles_sin_cos
    p[prefix + "torsion_angles_mask"] = torsion_angles_mask

    return p


# @curry1
def get_backbone_frames(p: TensorDict) -> TensorDict:
    # DISCREPANCY: AlphaFold uses tensor_7s here. I don't know why.

    p["backbone_rigid_tensor"] = p["rigidgroups_gt_frames"][..., 0, :, :]
    p["backbone_rigid_mask"] = p["rigidgroups_gt_exists"][..., 0]

    return p


# @curry1
def get_chi_angles(p: TensorDict) -> TensorDict:
    dtype = p["all_atom_mask"].dtype
    p["chi_angles_sin_cos"] = (p["torsion_angles_sin_cos"][..., 3:, :]).to(dtype)
    p["chi_mask"] = p["torsion_angles_mask"][..., 3:].to(dtype)

    return p


NUM_RES = "NUM_RES"


@curry1
def random_crop_to_size(
    p: TensorDict,
    crop_size: int,
    max_templates: int,
    shape_schema: Mapping[str, Sequence[Optional[str]]],
    subsample_templates: bool = False,
    seed: Optional[int] = None,
) -> TensorDict:
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    # We want each ensemble to be cropped the same way

    g = None
    if seed is not None:
        g = torch.Generator(device=p["seq_length"].device)
        g.manual_seed(seed)

    seq_length = p["seq_length"]

    if "template_mask" in p:
        num_templates = p["template_mask"].shape[-1]
    else:
        num_templates = 0

    # No need to subsample templates if there aren't any
    subsample_templates = subsample_templates and num_templates

    num_res_crop_size = min(int(seq_length), crop_size)

    def _randint(lower, upper):
        return int(torch.randint(lower, upper + 1, (1,), device=p["seq_length"].device, generator=g)[0])

    if subsample_templates:
        templates_crop_start = _randint(0, num_templates)
        templates_select_indices = torch.randperm(num_templates, device=p["seq_length"].device, generator=g)
    else:
        templates_crop_start = 0

    num_templates_crop_size = min(num_templates - templates_crop_start, max_templates)

    n = seq_length - num_res_crop_size
    if "use_clamped_fape" in p and p["use_clamped_fape"] == 1.0:
        right_anchor = n
    else:
        x = _randint(0, n)
        right_anchor = n - x

    num_res_crop_start = _randint(0, right_anchor)

    for k, v in p.items():
        if k not in shape_schema or ("template" not in k and NUM_RES not in shape_schema[k]):
            continue

        # Randomly permute the templates before cropping them.
        if k.startswith("template") and subsample_templates:
            v = v[templates_select_indices]

        slices = []
        for i, (dim_size, dim) in enumerate(zip(shape_schema[k], v.shape)):
            is_num_res = dim_size == NUM_RES
            if i == 0 and k.startswith("template"):
                crop_size = num_templates_crop_size
                crop_start = templates_crop_start
            else:
                crop_start = num_res_crop_start if is_num_res else 0
                crop_size = num_res_crop_size if is_num_res else dim
            slices.append(slice(crop_start, crop_start + crop_size))
        p[k] = v[slices]

    p["seq_length"] = p["seq_length"].new_tensor(num_res_crop_size)

    return p
