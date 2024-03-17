# Copyright 2024 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random
from typing import Dict

import torch

from deepfold.config import FEATURE_SHAPES, FeaturePipelineConfig
from deepfold.data import data_transforms

TensorDict = Dict[str, torch.Tensor]


def nonensembled_transform_fns(config: FeaturePipelineConfig):
    """Input pipeline data transformers that are not ensembled."""
    transforms = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.correct_msa_restypes,
        data_transforms.squeeze_features,
        data_transforms.randomly_replace_msa_with_unknown(0.0),
        data_transforms.make_seq_mask,
        data_transforms.make_msa_mask,
        data_transforms.make_hhblits_profile,
    ]
    if config.templates_enabled:
        transforms.extend(
            [
                data_transforms.fix_templates_aatype,
                data_transforms.make_template_mask,
                data_transforms.make_pseudo_beta("template_"),
            ]
        )
        if config.embed_template_torsion_angles:
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

    if config.supervised_features_enabled:
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
    config: FeaturePipelineConfig,
    ensemble_seed: int,
):
    """Input pipeline data transformers that can be ensembled and averaged."""
    transforms = []

    if config.block_delete_msa_enabled:
        transforms.append(
            data_transforms.block_delete_msa(
                msa_fraction_per_block=config.msa_fraction_per_deletion_block,
                randomize_num_blocks=config.randomize_num_msa_deletion_blocks,
                num_blocks=config.num_msa_deletion_blocks,
            )
        )

    if config.sample_msa_distillation_enabled:
        transforms.append(data_transforms.sample_msa_distillation(config.max_distillation_msa_clusters))

    if config.reduce_msa_clusters_by_max_templates:
        pad_msa_clusters = config.max_msa_clusters - config.max_templates
    else:
        pad_msa_clusters = config.max_msa_clusters

    max_msa_clusters = pad_msa_clusters
    max_extra_msa = config.max_extra_msa

    msa_seed = None
    if not config.resample_msa_in_recycling:
        msa_seed = ensemble_seed

    transforms.append(data_transforms.sample_msa(max_msa_clusters, keep_extra=True, seed=msa_seed))

    if config.masked_msa_enabled:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        transforms.append(
            data_transforms.make_masked_msa(
                profile_prob=config.masked_msa_profile_prob,
                same_prob=config.masked_msa_same_prob,
                uniform_prob=config.masked_msa_uniform_prob,
                replace_fraction=config.masked_msa_replace_fraction,
                seed=(msa_seed + 1) if msa_seed else None,
            )
        )

    if config.msa_cluster_features_enabled:
        transforms.append(data_transforms.nearest_neighbor_clusters())
        transforms.append(data_transforms.summarize_clusters())

    # Crop after creating the cluster profiles.
    if max_extra_msa:
        transforms.append(data_transforms.crop_extra_msa(max_extra_msa))
    else:
        transforms.append(data_transforms.delete_extra_msa)

    transforms.append(data_transforms.make_msa_feat())

    crop_feats = dict(FEATURE_SHAPES)

    if config.fixed_size:
        transforms.append(data_transforms.select_feat(list(crop_feats)))
        transforms.append(
            data_transforms.random_crop_to_size(
                config.crop_size,
                config.max_templates,
                crop_feats,
                config.subsample_templates,
                seed=ensemble_seed + 1,
            )
        )
        transforms.append(
            data_transforms.make_fixed_size(
                crop_feats,
                pad_msa_clusters,
                config.max_extra_msa,
                config.crop_size,
                config.max_templates,
            )
        )
    else:
        transforms.append(data_transforms.crop_templates(config.max_templates))

    return transforms


def process_tensors_from_config(
    tensors: TensorDict,
    config: FeaturePipelineConfig,
) -> TensorDict:
    """Based on the config, apply filters and transformations to the data."""

    # ensemble_seed = random.randint(0, torch.iinfo(torch.int32).max)
    ensemble_seed = config.ensemble_seed

    def wrap_ensemble_fn(data, i):
        """Function to be mapped over the ensemble dimension."""
        d = data.copy()
        fns = ensembled_transform_fns(config=config, ensemble_seed=ensemble_seed)
        fn = compose(fns)
        d["ensemble_index"] = i
        return fn(d)

    # no_templates = False
    # if "template_aatype" in tensors:
    #     no_templates = tensors["template_aatype"].shape[0] == 0

    nonensembled = nonensembled_transform_fns(config=config)

    tensors = compose(nonensembled)(tensors)

    if "num_recycling_iters" in tensors:
        num_recycling = int(tensors["num_recycling_iters"])
    else:
        num_recycling = config.max_recycling_iters

    tensors = map_fn(lambda x: wrap_ensemble_fn(tensors, x), torch.arange(num_recycling + 1))

    return tensors


@data_transforms.curry1
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
