# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited


import random
from typing import Dict

import torch
from omegaconf import DictConfig

from deepfold.model.v2.modules.input import data_ops

TensorDict = Dict[str, torch.Tensor]


def nonensembled_transform_fns(common_cfg: DictConfig, mode_cfg: DictConfig):
    """Input pipeline data transformers that are not ensembled."""
    transforms = [
        data_ops.cast_to_64bit_ints,
        data_ops.correct_msa_restypes,
        data_ops.squeeze_features,
        data_ops.randomly_replace_msa_with_unknown(0.0),
        data_ops.make_seq_mask,
        data_ops.make_msa_mask,
        data_ops.make_hhblits_profile,
    ]
    if common_cfg.use_templates:
        transforms.extend(
            [
                data_ops.fix_templates_aatype,
                data_ops.make_template_mask,
                data_ops.make_pseudo_beta("template_"),
            ]
        )
        if common_cfg.use_template_torsion_angles:
            transforms.extend(
                [
                    data_ops.atom37_to_torsion_angles("template_"),
                ]
            )

    transforms.extend(
        [
            data_ops.make_atom14_masks,
        ]
    )

    if mode_cfg.supervised:
        transforms.extend(
            [
                data_ops.make_atom14_positions,
                data_ops.atom37_to_frames,
                data_ops.atom37_to_torsion_angles(""),
                data_ops.make_pseudo_beta(""),
                data_ops.get_backbone_frames,
                data_ops.get_chi_angles,
            ]
        )

    return transforms


def ensembled_transform_fns(common_cfg: DictConfig, mode_cfg: DictConfig, ensemble_seed: int):
    """Input pipeline data transformers that can be ensembled and averaged."""
    transforms = []

    if mode_cfg.block_delete_msa:
        transforms.append(data_ops.block_delete_msa(common_cfg.block_delete_msa))

    if "max_distillation_msa_clusters" in mode_cfg:
        transforms.append(data_ops.sample_msa_distillation(mode_cfg.max_distillation_msa_clusters))

    if common_cfg.reduce_msa_clusters_by_max_templates:
        pad_msa_clusters = mode_cfg.max_msa_clusters - mode_cfg.max_templates
    else:
        pad_msa_clusters = mode_cfg.max_msa_clusters

    max_msa_clusters = pad_msa_clusters
    max_extra_msa = mode_cfg.max_extra_msa

    msa_seed = None
    if not common_cfg.resample_msa_in_recycling:
        msa_seed = ensemble_seed

    transforms.append(
        data_ops.sample_msa(
            max_msa_clusters,
            keep_extra=True,
            seed=msa_seed,
        )
    )

    if "masked_msa" in common_cfg:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        transforms.append(
            data_ops.make_masked_msa(
                common_cfg.masked_msa,
                mode_cfg.masked_msa_replace_fraction,
                seed=(msa_seed + 1) if msa_seed else None,
            )
        )

    if common_cfg.msa_cluster_features:
        transforms.append(data_ops.nearest_neighbor_clusters())
        transforms.append(data_ops.summarize_clusters())

    # Crop after creating the cluster profiles.
    if max_extra_msa:
        transforms.append(data_ops.crop_extra_msa(max_extra_msa))
    else:
        transforms.append(data_ops.delete_extra_msa)

    transforms.append(data_ops.make_msa_feat())

    crop_feats = dict(common_cfg.feat)

    if mode_cfg.fixed_size:
        transforms.append(data_ops.select_feat(list(crop_feats)))
        transforms.append(
            data_ops.random_crop_to_size(
                mode_cfg.crop_size,
                mode_cfg.max_templates,
                crop_feats,
                mode_cfg.subsample_templates,
                seed=ensemble_seed + 1,
            )
        )
        transforms.append(
            data_ops.make_fixed_size(
                crop_feats,
                pad_msa_clusters,
                mode_cfg.max_extra_msa,
                mode_cfg.crop_size,
                mode_cfg.max_templates,
            )
        )
    else:
        transforms.append(data_ops.crop_templates(mode_cfg.max_templates))

    return transforms


def process_tensors_from_config(tensors: TensorDict, common_cfg: DictConfig, mode_cfg: DictConfig):
    """Based on the config, apply filters and transformations to the data."""

    ensemble_seed = random.randint(0, torch.iinfo(torch.int32).max)

    def wrap_ensemble_fn(data, i):
        """Function to be mapped over the ensemble dimension."""
        d = data.copy()
        fns = ensembled_transform_fns(
            common_cfg,
            mode_cfg,
            ensemble_seed,
        )
        fn = compose(fns)
        d["ensemble_index"] = i
        return fn(d)

    no_templates = True
    if "template_aatype" in tensors:
        no_templates = tensors["template_aatype"].shape[0] == 0

    nonensembled = nonensembled_transform_fns(
        common_cfg,
        mode_cfg,
    )

    tensors = compose(nonensembled)(tensors)

    if "no_recycling_iters" in tensors:
        num_recycling = int(tensors["no_recycling_iters"])
    else:
        num_recycling = common_cfg.max_recycling_iters

    tensors = map_fn(lambda x: wrap_ensemble_fn(tensors, x), torch.arange(num_recycling + 1))

    return tensors


@data_ops.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x


def map_fn(fun, x):
    ensembles = [fun(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = torch.stack([dict_i[feat] for dict_i in ensembles], dim=-1)
    return ensembled_dict
