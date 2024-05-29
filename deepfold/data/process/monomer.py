# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 AlQuraishi Laboratory
# Copyright 2023 NVIDIA CORPORATION
# Copyright 2024 DeepFold Team


"""Feature proecessing logic for monomer data."""


from copy import deepcopy
from typing import Callable, Dict, List, Sequence

import torch

import deepfold.data.process.transforms as data_transforms
from deepfold.config import FEATURE_SHAPES, FeaturePipelineConfig

TensorDict = Dict[str, torch.Tensor]


def process_raw_feature_tensors(
    tensors: TensorDict,
    cfg: FeaturePipelineConfig,
) -> TensorDict:
    """Based on the config, apply filters and transformations to the data."""

    # nonensembled transformations:
    nonensembled = nonensembled_transform_fns(cfg)
    tensors = compose(nonensembled)(tensors)

    # ensembled transformations:
    ensembles = []
    for i in range(cfg.max_recycling_iters + 1):
        ensembled = ensembled_transform_fns(cfg, ensemble_iter=i)
        ensembles.append(compose(ensembled)(deepcopy(tensors)))

    tensors = {}
    for key in ensembles[0].keys():
        tensors[key] = torch.stack([d[key] for d in ensembles], dim=-1)

    return tensors


def nonensembled_transform_fns(cfg: FeaturePipelineConfig) -> List[Callable]:
    """Input pipeline data transformers that are not ensembled."""

    # Non-ensembled features:
    transforms = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.correct_msa_restypes,
        data_transforms.squeeze_features,
        data_transforms.randomly_replace_msa_with_unknown(0.0, cfg.seed),
        data_transforms.make_seq_mask,
        data_transforms.make_msa_mask,
        data_transforms.make_hhblits_profile,
    ]

    # Template features:
    if cfg.templates_enabled:
        transforms.extend(
            [
                data_transforms.fix_templates_aatype,
                data_transforms.make_template_mask,
                data_transforms.make_pseudo_beta("template_"),
            ]
        )
        if cfg.embed_template_torsion_angles:
            transforms.extend(
                [
                    data_transforms.atom37_to_torsion_angles("template_"),
                ]
            )

    transforms.extend(
        [
            data_transforms.make_atom14_masks,
        ]
    )

    # Supervised features:
    if cfg.supervised_features_enabled:
        transforms.extend(
            [
                data_transforms.make_atom14_positions,
                data_transforms.atom37_to_frames,
                data_transforms.atom37_to_torsion_angles(""),
                data_transforms.make_pseudo_beta(""),
                data_transforms.get_backbone_frames,
                data_transforms.get_chi_angles,
            ]
        )

    return transforms


def ensembled_transform_fns(
    cfg: FeaturePipelineConfig,
    ensemble_iter: int,
) -> List[Callable]:
    """Input pipeline data transformers that can be ensembled and averaged."""

    transforms = []

    if cfg.sample_msa_distillation_enabled:
        transforms.append(
            data_transforms.sample_msa_distillation(
                cfg.max_distillation_msa_clusters,
                (cfg.seed + ensemble_iter),
            )
        )

    transforms.append(
        data_transforms.sample_msa(
            cfg.max_msa_clusters,
            keep_extra=True,
            seed=(cfg.seed + ensemble_iter),
        )
    )

    if cfg.masked_msa_enabled:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        transforms.append(
            data_transforms.make_masked_msa(
                cfg.masked_msa_profile_prob,
                cfg.masked_msa_same_prob,
                cfg.masked_msa_uniform_prob,
                cfg.masked_msa_replace_fraction,
                (cfg.seed + ensemble_iter),
            )
        )

    if cfg.msa_cluster_features_enabled:
        transforms.append(data_transforms.nearest_neighbor_clusters())
        transforms.append(data_transforms.summarize_clusters())

    # Crop after creating the cluster profiles.
    if cfg.max_extra_msa:
        transforms.append(
            data_transforms.crop_extra_msa(
                cfg.max_extra_msa,
                (cfg.seed + ensemble_iter),
            )
        )
    else:
        transforms.append(data_transforms.delete_extra_msa)

    transforms.append(data_transforms.make_msa_feat())

    transforms.append(
        data_transforms.filter_features(
            allowed_feature_names=set(FEATURE_SHAPES.keys()),
        )
    )

    transforms.append(
        data_transforms.random_crop_and_template_subsampling(
            feature_schema_shapes=FEATURE_SHAPES,
            sequence_crop_size=cfg.crop_size,
            max_templates=cfg.max_templates,
            subsample_templates=cfg.subsample_templates,
            seed=cfg.seed,
        )
    )

    if cfg.fixed_size:
        transforms.append(
            data_transforms.pad_to_schema_shape(
                feature_schema_shapes=FEATURE_SHAPES,
                num_residues=cfg.crop_size,
                num_clustered_msa_seq=cfg.max_msa_clusters,
                num_extra_msa_seq=cfg.max_extra_msa,
                num_templates=cfg.max_templates,
            )
        )

    return transforms


@data_transforms.curry1
def compose(
    x: TensorDict,
    fs: Sequence[Callable],
) -> TensorDict:
    for f in fs:
        x = f(x)
    return x
