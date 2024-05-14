# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 AlQuraishi Laboratory
# Copyright 2023 NVIDIA CORPORATION
# Copyright 2024 DeepFold Team


import time
from copy import deepcopy
from typing import Callable, Dict, List, Sequence

import torch

import deepfold.data.process.transforms as data_transforms
from deepfold.config import FEATURE_SHAPES, FeaturePipelineConfig


def process_raw_feature_tensors(
    tensors: Dict[str, torch.Tensor],
    pipeline_config: FeaturePipelineConfig,
    mode: str,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Based on the config, apply filters and transformations to the data."""

    if mode == "train":
        sequence_crop_size = pipeline_config.crop_size
    elif mode in {"eval", "predict"}:
        sequence_crop_size = tensors["seq_length"][0].item()

    # nonensembled transformations:
    _compose_nonensembled_perf = -time.perf_counter()
    nonensembled = nonensembled_transform_fns(
        pipeline_config=pipeline_config,
        mode=mode,
        seed=seed,
    )
    tensors = compose(nonensembled)(tensors)
    _compose_nonensembled_perf += time.perf_counter()

    # ensembled transformations:
    _compose_ensembled_perf = -time.perf_counter()
    ensembles = []
    for i in range(pipeline_config.num_recycling_iters + 1):
        ensembled = ensembled_transform_fns(
            pipeline_config=pipeline_config,
            sequence_crop_size=sequence_crop_size,
            mode=mode,
            seed=seed,
            ensemble_iter=i,
        )
        ensembles.append(compose(ensembled)(deepcopy(tensors)))
    tensors = {}
    for key in ensembles[0].keys():
        tensors[key] = torch.stack([d[key] for d in ensembles], dim=-1)
    _compose_ensembled_perf += time.perf_counter()

    return tensors


def nonensembled_transform_fns(
    pipeline_config: FeaturePipelineConfig,
    mode: str,
    seed: int,
) -> List[Callable]:
    """Input pipeline data transformers that are not ensembled."""

    transforms = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.correct_msa_restypes,
        data_transforms.squeeze_features,
        data_transforms.randomly_replace_msa_with_unknown(0.0, seed),
        data_transforms.make_seq_mask,
        data_transforms.make_msa_mask,
        data_transforms.make_hhblits_profile,
    ]
    if pipeline_config.templates_enabled:
        transforms.extend(
            [
                data_transforms.fix_templates_aatype,
                data_transforms.make_template_mask,
                data_transforms.make_pseudo_beta("template_"),
            ]
        )
        if pipeline_config.embed_template_torsion_angles:
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

    if mode in {"train", "eval"}:
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
    pipeline_config: FeaturePipelineConfig,
    sequence_crop_size: int,
    mode: str,
    seed: int,
    ensemble_iter: int,
) -> List[Callable]:
    """Input pipeline data transformers that can be ensembled and averaged."""

    transforms = []

    if mode == "train":
        transforms.append(
            data_transforms.sample_msa_distillation(
                pipeline_config.max_distillation_msa_clusters,
                (seed + ensemble_iter),
            )
        )

    transforms.append(
        data_transforms.sample_msa(
            pipeline_config.max_msa_clusters,
            keep_extra=True,
            seed=(seed + ensemble_iter),
        )
    )

    if pipeline_config.masked_msa_enabled:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        transforms.append(
            data_transforms.make_masked_msa(
                pipeline_config.masked_msa_profile_prob,
                pipeline_config.masked_msa_same_prob,
                pipeline_config.masked_msa_uniform_prob,
                pipeline_config.masked_msa_replace_fraction,
                (seed + ensemble_iter),
            )
        )

    if pipeline_config.msa_cluster_features:
        transforms.append(data_transforms.nearest_neighbor_clusters())
        transforms.append(data_transforms.summarize_clusters())

    # Crop after creating the cluster profiles.
    if pipeline_config.max_extra_msa:
        transforms.append(
            data_transforms.crop_extra_msa(
                pipeline_config.max_extra_msa,
                (seed + ensemble_iter),
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

    if mode == "train":
        subsample_templates = True
    elif mode in {"eval", "predict"}:
        subsample_templates = False

    transforms.append(
        data_transforms.random_crop_and_template_subsampling(
            feature_schema_shapes=FEATURE_SHAPES,
            sequence_crop_size=sequence_crop_size,
            max_templates=pipeline_config.max_templates,
            subsample_templates=subsample_templates,
            seed=seed,
        )
    )
    transforms.append(
        data_transforms.pad_to_schema_shape(
            feature_schema_shapes=FEATURE_SHAPES,
            num_residues=sequence_crop_size,
            num_clustered_msa_seq=pipeline_config.max_msa_clusters,
            num_extra_msa_seq=pipeline_config.max_extra_msa,
            num_templates=pipeline_config.max_templates,
        )
    )

    return transforms


@data_transforms.curry1
def compose(
    x: dict,
    fs: Sequence[Callable],
) -> dict:
    for f in fs:
        x = f(x)
    return x
