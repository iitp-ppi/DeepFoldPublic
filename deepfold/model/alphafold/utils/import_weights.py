# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited

import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import List, Union

import numpy as np
import torch
from omegaconf import DictConfig

from deepfold.model.alphafold.model import AlphaFold
from deepfold.model.alphafold.nn.evoformer import EvoformerBlock

logger = logging.getLogger(__name__)


_NPZ_KEY_PREFIX = "alphafold/alphafold_iteration/"


# With Param, a poor man's enum with attributes (Rust-style)
class ParamType(Enum):
    LinearWeight = partial(lambda w: w.transpose(-1, -2))  # hack: partial prevents fns from becoming methods
    LinearWeightMHA = partial(lambda w: w.reshape(*w.shape[:-2], -1).transpose(-1, -2))
    LinearMHAOutputWeight = partial(lambda w: w.reshape(*w.shape[:-3], -1, w.shape[-1]).transpose(-1, -2))
    LinearBiasMHA = partial(lambda w: w.reshape(*w.shape[:-2], -1))
    LinearWeightOPM = partial(lambda w: w.reshape(*w.shape[:-3], -1, w.shape[-1]).transpose(-1, -2))
    Other = partial(lambda w: w)

    def __init__(self, fn):
        self.transformation = fn


@dataclass
class Param:
    param: Union[torch.Tensor, List[torch.Tensor]]
    param_type: ParamType = ParamType.Other
    stacked: bool = False


def process_translation_dict(d, top_layer=True):
    flat = {}
    for k, v in d.items():
        if type(v) == dict:
            prefix = _NPZ_KEY_PREFIX if top_layer else ""
            sub_flat = {(prefix + "/".join([k, k_prime])): v_prime for k_prime, v_prime in process_translation_dict(v, top_layer=False).items()}
            flat.update(sub_flat)
        else:
            k = "/" + k if not top_layer else k
            flat[k] = v

    return flat


def stacked(param_dict_list, out=None):
    """
    Args:
        param_dict_list:
            A list of (nested) Param dicts to stack. The structure of
            each dict must be the identical (down to the ParamTypes of
            "parallel" Params). There must be at least one dict
            in the list.
    """
    if out is None:
        out = {}
    template = param_dict_list[0]
    for k, _ in template.items():
        v = [d[k] for d in param_dict_list]
        if type(v[0]) is dict:
            out[k] = {}
            stacked(v, out=out[k])
        elif type(v[0]) is Param:
            stacked_param = Param(
                param=[param.param for param in v],
                param_type=v[0].param_type,
                stacked=True,
            )

            out[k] = stacked_param

    return out


def assign(translation_dict, orig_weights):
    for k, param in translation_dict.items():
        with torch.no_grad():
            weights = torch.as_tensor(orig_weights[k])
            ref, param_type = param.param, param.param_type
            if param.stacked:
                weights = torch.unbind(weights, 0)
            else:
                weights = [weights]
                ref = [ref]

            try:
                weights = list(map(param_type.transformation, weights))
                for p, w in zip(ref, weights):
                    p.copy_(w)
            except:
                logger.debug(f"{k}: {tuple(ref[0].shape)} != {tuple(weights[0].shape)}")
                raise


#######################
# Primitive templates
#######################

# Linear layer

LinearWeight = lambda l: (Param(l, param_type=ParamType.LinearWeight))

LinearBias = lambda l: (Param(l))

# Multi-Head Attention

LinearWeightMHA = lambda l: (Param(l, param_type=ParamType.LinearWeightMHA))

LinearBiasMHA = lambda b: (Param(b, param_type=ParamType.LinearBiasMHA))

# Outer Product Mean

LinearWeightOPM = lambda l: (Param(l, param_type=ParamType.LinearWeightOPM))

#######################
# Some templates
#######################

LinearParams = lambda l: {
    "weights": LinearWeight(l.weight),
    "bias": LinearBias(l.bias),
}

LayerNormParams = lambda l: {
    "scale": Param(l.weight),
    "offset": Param(l.bias),
}

AttentionParams = lambda att: {
    "query_w": LinearWeightMHA(att.linear_q.weight),
    "key_w": LinearWeightMHA(att.linear_k.weight),
    "value_w": LinearWeightMHA(att.linear_v.weight),
    "output_w": Param(
        att.linear_o.weight,
        param_type=ParamType.LinearMHAOutputWeight,
    ),
    "output_b": LinearBias(att.linear_o.bias),
}

AttentionGatedParams = lambda att: dict(
    **AttentionParams(att),
    **{
        "gating_w": LinearWeightMHA(att.linear_g.weight),
        "gating_b": LinearBiasMHA(att.linear_g.bias),
    },
)

GlobalAttentionParams = lambda att: dict(
    AttentionGatedParams(att),
    key_w=LinearWeight(att.linear_k.weight),
    value_w=LinearWeight(att.linear_v.weight),
)

TriAttParams = lambda tri_att: {
    "query_norm": LayerNormParams(tri_att.layer_norm),
    "feat_2d_weights": LinearWeight(tri_att.linear.weight),
    "attention": AttentionGatedParams(tri_att.mha),
}

TriMulOutParams = lambda tri_mul: {
    "layer_norm_input": LayerNormParams(tri_mul.layer_norm_in),
    "left_projection": LinearParams(tri_mul.linear_a_p),
    "right_projection": LinearParams(tri_mul.linear_b_p),
    "left_gate": LinearParams(tri_mul.linear_a_g),
    "right_gate": LinearParams(tri_mul.linear_b_g),
    "center_layer_norm": LayerNormParams(tri_mul.layer_norm_out),
    "output_projection": LinearParams(tri_mul.linear_z),
    "gating_linear": LinearParams(tri_mul.linear_g),
}

# see commit b88f8da on the Alphafold repo
# Alphafold swaps the pseudocode's a and b between the incoming/outcoming
# iterations of triangle multiplication, which is confusing and not
# reproduced in our implementation.
TriMulInParams = lambda tri_mul: {
    "layer_norm_input": LayerNormParams(tri_mul.layer_norm_in),
    "left_projection": LinearParams(tri_mul.linear_b_p),
    "right_projection": LinearParams(tri_mul.linear_a_p),
    "left_gate": LinearParams(tri_mul.linear_b_g),
    "right_gate": LinearParams(tri_mul.linear_a_g),
    "center_layer_norm": LayerNormParams(tri_mul.layer_norm_out),
    "output_projection": LinearParams(tri_mul.linear_z),
    "gating_linear": LinearParams(tri_mul.linear_g),
}

PairTransitionParams = lambda pt: {
    "input_layer_norm": LayerNormParams(pt.layer_norm),
    "transition1": LinearParams(pt.linear_1),
    "transition2": LinearParams(pt.linear_2),
}

MSAAttParams = lambda matt: {
    "query_norm": LayerNormParams(matt.layer_norm_m),
    "attention": AttentionGatedParams(matt.mha),
}

MSAColAttParams = lambda matt: {
    "query_norm": LayerNormParams(matt._msa_att.layer_norm_m),
    "attention": AttentionGatedParams(matt._msa_att.mha),
}

MSAGlobalAttParams = lambda matt: {
    "query_norm": LayerNormParams(matt.layer_norm_m),
    "attention": GlobalAttentionParams(matt.global_attention),
}

MSAAttPairBiasParams = lambda matt: dict(
    **MSAAttParams(matt),
    **{
        "feat_2d_norm": LayerNormParams(matt.layer_norm_z),
        "feat_2d_weights": LinearWeight(matt.linear_z.weight),
    },
)

IPAParams = lambda ipa: {
    "q_scalar": LinearParams(ipa.linear_q),
    "kv_scalar": LinearParams(ipa.linear_kv),
    "q_point_local": LinearParams(ipa.linear_q_points),
    "kv_point_local": LinearParams(ipa.linear_kv_points),
    "trainable_point_weights": Param(param=ipa.head_weights, param_type=ParamType.Other),
    "attention_2d": LinearParams(ipa.linear_b),
    "output_projection": LinearParams(ipa.linear_out),
}

TemplatePairBlockParams = lambda b: {
    "triangle_attention_starting_node": TriAttParams(b.tri_att_start),
    "triangle_attention_ending_node": TriAttParams(b.tri_att_end),
    "triangle_multiplication_outgoing": TriMulOutParams(b.tri_mul_out),
    "triangle_multiplication_incoming": TriMulInParams(b.tri_mul_in),
    "pair_transition": PairTransitionParams(b.pair_transition),
}

MSATransitionParams = lambda m: {
    "input_layer_norm": LayerNormParams(m.layer_norm),
    "transition1": LinearParams(m.linear_1),
    "transition2": LinearParams(m.linear_2),
}

OuterProductMeanParams = lambda o: {
    "layer_norm_input": LayerNormParams(o.layer_norm),
    "left_projection": LinearParams(o.linear_1),
    "right_projection": LinearParams(o.linear_2),
    "output_w": LinearWeightOPM(o.linear_out.weight),
    "output_b": LinearBias(o.linear_out.bias),
}


def EvoformerBlockParams(b: EvoformerBlock, is_extra_msa=False):
    if is_extra_msa:
        col_att_name = "msa_column_global_attention"
        msa_col_att_params = MSAGlobalAttParams(b.msa.msa_att_col)
    else:
        col_att_name = "msa_column_attention"
        msa_col_att_params = MSAColAttParams(b.msa.msa_att_col)

    d = {
        "msa_row_attention_with_pair_bias": MSAAttPairBiasParams(b.msa.msa_att_row),
        col_att_name: msa_col_att_params,
        "msa_transition": MSATransitionParams(b.msa.msa_transition),
        "outer_product_mean": OuterProductMeanParams(b.communication),
        "triangle_multiplication_outgoing": TriMulOutParams(b.pair.tri_mul_out),
        "triangle_multiplication_incoming": TriMulInParams(b.pair.tri_mul_in),
        "triangle_attention_starting_node": TriAttParams(b.pair.tri_att_start),
        "triangle_attention_ending_node": TriAttParams(b.pair.tri_att_end),
        "pair_transition": PairTransitionParams(b.pair.pair_transition),
    }

    return d


ExtraMSABlockParams = partial(EvoformerBlockParams, is_extra_msa=True)

FoldIterationParams = lambda sm: {
    "invariant_point_attention": IPAParams(sm.ipa),
    "attention_layer_norm": LayerNormParams(sm.layer_norm_ipa),
    "transition": LinearParams(sm.transition.layers[0].linear_1),
    "transition_1": LinearParams(sm.transition.layers[0].linear_2),
    "transition_2": LinearParams(sm.transition.layers[0].linear_3),
    "transition_layer_norm": LayerNormParams(sm.transition.layer_norm),
    "affine_update": LinearParams(sm.bb_update.linear),
    "rigid_sidechain": {
        "input_projection": LinearParams(sm.angle_resnet.linear_in),
        "input_projection_1": LinearParams(sm.angle_resnet.linear_initial),
        "resblock1": LinearParams(sm.angle_resnet.layers[0].linear_1),
        "resblock2": LinearParams(sm.angle_resnet.layers[0].linear_2),
        "resblock1_1": LinearParams(sm.angle_resnet.layers[1].linear_1),
        "resblock2_1": LinearParams(sm.angle_resnet.layers[1].linear_2),
        "unnormalized_angles": LinearParams(sm.angle_resnet.linear_out),
    },
}


def generate_translation_dict(
    model: AlphaFold,
    *,
    enable_template: bool = True,
    enable_ptm: bool = False,
):
    ############################
    # Translations dict overflow
    ############################

    ems_blocks = model.extra_msa_stack.blocks
    ems_blocks_params = stacked([ExtraMSABlockParams(b) for b in ems_blocks])

    evo_blocks = model.evoformer.blocks
    evo_blocks_params = stacked([EvoformerBlockParams(b) for b in evo_blocks])

    translations = {
        "evoformer": {
            "preprocess_1d": LinearParams(model.input_embedder.linear_tf_m),
            "preprocess_msa": LinearParams(model.input_embedder.linear_msa_m),
            "left_single": LinearParams(model.input_embedder.linear_tf_z_i),
            "right_single": LinearParams(model.input_embedder.linear_tf_z_j),
            "prev_pos_linear": LinearParams(model.recycling_embedder.linear),
            "prev_msa_first_row_norm": LayerNormParams(model.recycling_embedder.layer_norm_m),
            "prev_pair_norm": LayerNormParams(model.recycling_embedder.layer_norm_z),
            "pair_activiations": LinearParams(model.input_embedder.linear_relpos),
            "extra_msa_activations": LinearParams(model.extra_msa_embedder.linear),
            "extra_msa_stack": ems_blocks_params,
            "evoformer_iteration": evo_blocks_params,
            "single_activations": LinearParams(model.evoformer.linear),
        },
        "structure_module": {
            "single_layer_norm": LayerNormParams(model.structure_module.layer_norm_s),
            "initial_projection": LinearParams(model.structure_module.linear_in),
            "pair_layer_norm": LayerNormParams(model.structure_module.layer_norm_z),
            "fold_iteration": FoldIterationParams(model.structure_module),
        },
        "predicted_lddt_head": {
            "input_layer_norm": LayerNormParams(model.aux_heads.plddt.layer_norm),
            "act_0": LinearParams(model.aux_heads.plddt.linear_1),
            "act_1": LinearParams(model.aux_heads.plddt.linear_2),
            "logits": LinearParams(model.aux_heads.plddt.linear_3),
        },
        "distogram_head": {
            "half_logits": LinearParams(model.aux_heads.distogram.linear),
        },
        "experimentally_resolved_head": {
            "logits": LinearParams(model.aux_heads.experimentally_resolved.linear),
        },
        "masked_msa_head": {
            "logits": LinearParams(model.aux_heads.masked_msa.linear),
        },
    }

    if enable_template:
        tps_blocks = model.template_pair_stack.blocks
        tps_blocks_params = stacked([TemplatePairBlockParams(b) for b in tps_blocks])
        template_param_dict = {
            "template_embedding": {
                "single_template_embedding": {
                    "embedding2d": LinearParams(model.template_pair_embedder.linear),
                    "template_pair_stack": {
                        "__layer_stack_no_state": tps_blocks_params,
                    },
                    "output_layer_norm": LayerNormParams(model.template_pair_stack.layer_norm),
                },
                "attention": AttentionParams(model.template_pointwise_att.mha),
            },
            "template_single_embedding": LinearParams(model.template_angle_embedder.linear_1),
            "template_projection": LinearParams(model.template_angle_embedder.linear_2),
        }

        translations["evoformer"].update(template_param_dict)

    if enable_ptm:
        translations["predicted_aligned_error_head"] = {"logits": LinearParams(model.aux_heads.tm.linear)}

    return translations


def import_jax_weights_(
    model: torch.nn.Module,
    npz_path: Union[str, bytes, os.PathLike],
    config: DictConfig,
):
    # Load jax params
    data = np.load(npz_path)

    # Check template usage
    num_templ = [
        "model_3",
        "model_4",
        "model_5",
        "model_3_ptm",
        "model_4_ptm",
        "model_5_ptm",
    ]
    use_template = config.globals.use_template

    # Check ptm usage
    use_ptm = config.globals.tm_enabled

    # Generate translation dictionary
    translations = generate_translation_dict(
        model,
        enable_template=use_template,
        enable_ptm=use_ptm,
    )

    # Flatten keys and insert missing key prefixes
    flat = process_translation_dict(translations)

    # Sanity check
    keys = list(data.keys())
    flat_keys = list(flat.keys())
    incorrect = [k for k in flat_keys if k not in keys]
    missing = [k for k in keys if k not in flat_keys]

    # for x in incorrect:
    #     logger.debug(f"Incorrect: {x}")

    # for x in missing:
    #     logger.debug(f"Missing: {x}")

    assert len(incorrect) == 0
    # assert(sorted(list(flat.keys())) == sorted(list(data.keys())))

    # Set weights
    assign(flat, data)
