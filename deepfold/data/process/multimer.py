# Copyright 2021 DeepMind Technologies Limited
# Copyright 2024 DeepFold Team


"""Feature processing logic for multimer data."""


from copy import deepcopy
from typing import Callable, Dict, List

import torch

from deepfold.config import MULTIMER_FEATURE_SHAPES, FeaturePipelineConfig
from deepfold.data.process import transforms as data_transforms
from deepfold.data.process import transforms_multimer as data_transforms_multimer
from deepfold.data.process.monomer import compose

TensorDict = Dict[str, torch.Tensor]


def process_raw_feature_tensors(
    tensors: TensorDict,
    cfg: FeaturePipelineConfig,
) -> TensorDict:
    """Based on the config, apply filters and transformations to the data."""

    # nonesnsembled transformations:
    # tensors["aatype"] = tensors["aatype"].to(torch.long)
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
    """Input pipeline data transfroms that are not ensembled."""

    # Non-ensembled fetaures:
    transforms = [
        data_transforms.cast_to_64bit_ints,
        data_transforms_multimer.make_msa_profile,
        data_transforms_multimer.create_target_feat,
    ]

    # Template features:
    transforms.extend(
        [
            data_transforms.make_pseudo_beta("template_"),
            data_transforms.atom37_to_torsion_angles("template_"),
        ]
    )
    # TODO: multichain_2d_mask

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

    transforms.extend(
        [
            data_transforms.make_atom14_masks,
        ]
    )

    return transforms


def ensembled_transform_fns(
    cfg: FeaturePipelineConfig,
    ensemble_iter: int,
) -> List[Callable]:
    """Input pipeline data transformers that can be ensembled and averaged."""

    transforms = []

    pad_msa_clusters = cfg.max_extra_msa
    max_msa_clusters = pad_msa_clusters
    max_extra_msa = cfg.max_extra_msa

    msa_seed = cfg.seed
    if cfg.resample_msa_in_recycling:
        msa_seed = cfg.seed + ensemble_iter

    transforms.append(
        data_transforms_multimer.sample_msa(
            max_msa_clusters,
            max_extra_msa,
            seed=msa_seed,
        )
    )

    if cfg.masked_msa_enabled:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        transforms.append(
            data_transforms_multimer.make_masked_msa(
                cfg.masked_msa_profile_prob,
                cfg.masked_msa_same_prob,
                cfg.masked_msa_uniform_prob,
                cfg.masked_msa_replace_fraction,
                seed=msa_seed,
            )
        )

    if cfg.msa_cluster_features_enabled:
        transforms.append(data_transforms_multimer.nearest_neighbor_clusters(gap_agreement_weight=0.0))

    transforms.append(data_transforms_multimer.create_msa_feat)

    transforms.append(data_transforms.filter_features(allowed_feature_names=set(MULTIMER_FEATURE_SHAPES.keys())))

    transforms.append(
        data_transforms_multimer.random_crop_and_template_subsampling(
            feature_schema_shapes=MULTIMER_FEATURE_SHAPES,
            sequence_crop_size=cfg.crop_size,
            max_templates=cfg.max_templates,
            subsample_templates=cfg.subsample_templates,
            spatial_crop_prob=cfg.spatial_crop_prob,
            interface_threshold=cfg.interface_threshold,
            seed=msa_seed,
        )
    )

    if cfg.fixed_size:
        transforms.append(
            data_transforms.pad_to_schema_shape(
                feature_schema_shapes=MULTIMER_FEATURE_SHAPES,
                num_residues=cfg.crop_size,
                num_clustered_msa_seq=cfg.max_msa_clusters,
                num_extra_msa_seq=cfg.max_extra_msa,
                num_templates=cfg.max_templates,
            )
        )

    return transforms
