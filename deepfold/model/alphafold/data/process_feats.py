# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited


"""Code to generate processed features."""


import random
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

import deepfold.model.alphafold.data.transforms as dt
from deepfold.model.alphafold.data.types import FeatureDict, TensorDict


def np_to_tensor_dict(
    np_example: Mapping[str, np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
) -> TensorDict:
    """
    Convert a dict of ndarray features to dict of (filtered) tensors.
    """

    allow_types = (
        np.float64,
        np.float32,
        np.float16,
        np.complex64,
        np.complex128,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint8,
        np.bool_,
    )
    to_tensor = lambda a: torch.tensor(a) if type(a) != torch.Tensor else a.clone().detach()
    tensor_dict = {
        k: to_tensor(v)
        for k, v in np_example.items()
        if (feature_names is not None) and (k in feature_names) and (v.dtype in allow_types)
    }

    return tensor_dict


def make_data_config(cfg: DictConfig, mode: str, num_res: int) -> Tuple[DictConfig, List[str]]:
    """
    Make a data config for the input pipeline.

    Args:
        cfg: omegaconf.DictConfig
            Configuration dictionary.
        mode: str
            Data mode: predict, eval, train
        num_res: int
            Number of residues
    """

    cfg = cfg.copy()
    mode_cfg = cfg[mode]
    if mode_cfg.crop_size is None:
        mode_cfg.crop_size = num_res

    feature_names = [str(name) for name in cfg.feat.unsupervised_features]

    if cfg.feat.use_template:
        feature_names += [str(name) for name in cfg.feat.template_features]

    if cfg[mode].supervised:
        feature_names += [str(name) for name in cfg.feat.supervised_features]

    return cfg, feature_names


def np_example_to_features(np_example: FeatureDict, cfg: DictConfig, mode: str) -> TensorDict:
    """
    Generate processed features.
    """

    np_example = dict(np_example)
    num_res = int(np_example["seq_length"][0])
    cfg, feature_names = make_data_config(cfg=cfg, mode=mode, num_res=num_res)

    if "deletion_matrix_int" in np_example:
        np_example["deletion_matrix"] = np_example.pop("deletion_matrix_int").astype(np.float32)

    tensor_dict = np_to_tensor_dict(np_example=np_example, feature_names=feature_names)

    with torch.no_grad():
        features = process_tensors_from_config(tensor_dict, cfg.feat, cfg[mode])

    if mode == "train":
        p = torch.rand(1).item()
        use_clamped_fape_value = float(p < cfg.train.clamp_prob)
        features["use_clamped_fape"] = torch.full(
            size=[cfg.feat.max_recycling_iters + 1],
            fill_value=use_clamped_fape_value,
            dtype=torch.float32,
        )
    else:  # eval, predict
        features["use_clamped_fape"] = torch.full(
            size=[cfg.feat.max_recycling_iters + 1],
            fill_value=0.0,
            dtype=torch.float32,
        )

    return {k: v for k, v in features.items()}


def nonensembled_transform_fns(
    feat_cfg: DictConfig,  # cfg.data.feat
    mode_cfg: DictConfig,  # cfg.data.<MODE>
) -> List[dt.TransformFn]:
    """Input pipeline data transformers that are not ensembled."""

    transforms = [
        dt.cast_to_int64,
        dt.correct_msa_restypes,
        dt.squeeze_features,
        dt.randomly_replace_msa_with_unknown(0.0),
        dt.make_seq_mask,
        dt.make_msa_mask,
        dt.make_hhblits_profile,
    ]
    if feat_cfg.use_templates:
        transforms.extend(
            [
                dt.fix_templates_aatype,
                dt.make_template_mask,
                dt.make_pseudo_beta("template_"),
            ]
        )
        if feat_cfg.use_template_torsion_angles:
            transforms.extend(
                [
                    dt.atom37_to_torsion_angles("template_"),
                ]
            )

    transforms.extend(
        [
            dt.make_atom14_masks,
        ]
    )

    if mode_cfg.supervised:
        transforms.extend(
            [
                dt.make_atom14_positions,
                dt.atom37_to_frames,
                dt.atom37_to_torsion_angles(""),
                dt.make_pseudo_beta(""),
                dt.get_backbone_frames,
                dt.get_chi_angles,
            ]
        )

    return transforms


def ensembled_transform_fns(
    feat_cfg: DictConfig,  # cfg.data.feat
    mode_cfg: DictConfig,  # cfg.data.<MODE>
    ensemble_seed: int,
) -> List[dt.TransformFn]:
    """Input pipeline data transformers that can be ensembled and averaged."""

    transforms = []

    if "max_distillation_msa_clusters" in mode_cfg:
        transforms.append(dt.sample_msa_distillation(mode_cfg.max_distillation_msa_clusters))

    if feat_cfg.reduce_msa_clusters_by_max_templates:
        pad_msa_clusters = mode_cfg.max_msa_clusters - mode_cfg.max_templates
    else:
        pad_msa_clusters = mode_cfg.max_msa_clusters

    max_msa_clusters = pad_msa_clusters
    max_extra_msa = mode_cfg.max_extra_msa

    msa_seed = None
    if not feat_cfg.resample_msa_in_recycling:
        msa_seed = ensemble_seed

    transforms.append(dt.sample_msa(max_msa_clusters, keep_extra=True, seed=msa_seed))

    if "masked_msa" in feat_cfg:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        transforms.append(dt.make_masked_msa(feat_cfg.masked_msa, mode_cfg.masked_msa_replace_fraction))

    if feat_cfg.msa_cluster_features:
        transforms.append(dt.nearest_neighbor_clusters())
        transforms.append(dt.summarize_clusters())

    # Crop after creating the cluster profiles.
    if max_extra_msa:
        transforms.append(dt.crop_extra_msa(max_extra_msa))
    else:
        transforms.append(dt.delete_extra_msa)

    transforms.append(dt.make_msa_feat())

    crop_feats = dict(feat_cfg.features)

    if mode_cfg.fixed_size:
        transforms.append(dt.select_features(list(crop_feats)))
        transforms.append(
            dt.random_crop_to_size(
                mode_cfg.crop_size,
                mode_cfg.max_templates,
                crop_feats,
                mode_cfg.subsample_templates,
                seed=ensemble_seed + 1,
            )
        )
        transforms.append(
            dt.make_fixed_size(
                crop_feats,
                pad_msa_clusters,
                mode_cfg.max_extra_msa,
                mode_cfg.crop_size,
                mode_cfg.max_templates,
            )
        )
    else:
        transforms.append(dt.crop_templates(mode_cfg.max_templates))

    return transforms


def process_tensors_from_config(
    feats: TensorDict,
    feat_cfg: DictConfig,  # cfg.data.feat
    mode_cfg: DictConfig,  # cfg.data.<MODE>
) -> TensorDict:
    """Based on the config, apply filters and transformations to the data."""

    ensemble_seed = random.randint(0, torch.iinfo(torch.int32).max)

    def wrap_ensemble_fn(data, i):
        """Function to be mapped over the ensemble dimension."""
        d = data.copy()
        fns = ensembled_transform_fns(feat_cfg, mode_cfg, ensemble_seed)
        fn = dt.compose(fns)
        d["ensemble_index"] = i
        return fn(d)

    nonensembled = nonensembled_transform_fns(feat_cfg, mode_cfg)

    feats = dt.compose(nonensembled)(feats)

    if "num_recycling_iters" in feats:
        num_recycling = int(feats["num_recycling_iters"])
    else:
        num_recycling = feat_cfg.max_recycling_iters

    feats = dt.map_fn(lambda x: wrap_ensemble_fn(feats, x), torch.arange(num_recycling + 1))

    return feats
