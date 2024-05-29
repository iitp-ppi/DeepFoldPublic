# Copyright 2024 DeepFold Team
# Copyright 2022 DP Technology
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited


import logging
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import List, Union

import numpy as np
import torch

from deepfold.modules.alphafold import AlphaFold

logger = logging.getLogger(__name__)

_NPZ_KEY_PREFIX = "alphafold/alphafold_iteration/"


def reshape_weight(x: np.ndarray) -> np.ndarray:
    len_shape = len(x.shape)
    if len_shape == 2:
        return x.transpose(-1, -2)
    elif len_shape == 1:
        return x.reshape(-1, 1)
    else:
        raise RuntimeError("Wrong shape")


# NOTE: `partial`` prevents `fns` from becoming methods.
class ParamType(Enum):
    LinearWeight = partial(lambda w: reshape_weight(w))
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
    swap: bool = False


def _process_translations_dict(d, top_layer=True):
    flat = {}
    for k, v in d.items():
        if type(v) == dict:
            prefix = _NPZ_KEY_PREFIX if top_layer else ""
            sub_flat = {(prefix + "/".join([k, k_prime])): v_prime for k_prime, v_prime in _process_translations_dict(v, top_layer=False).items()}
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
                swap=v[0].swap,
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
                    if param.swap:
                        index = p.shape[0] // 2
                        p[:index].copy_(w[index:])
                        p[index:].copy_(w[:index])
                    else:
                        p.copy_(w)
            except:
                logger.debug(f"{k}: Incorrect translation from {ref[0].shape} to {weights[0].shape}")
                print(k)
                print(ref[0].shape)
                print(weights[0].shape)
                raise


def import_jax_weights_(
    model: AlphaFold,
    npz_path: str,
    is_multimer: bool = False,
    enable_ptm: bool = False,
    enable_templates: bool = False,
    fuse_projection_weights: bool = False,
) -> None:
    """Import AlphaFold JAX parameters.

    Args:
        model: AlphaFold model.
        npz_path: Path to an NPZ file.
        is_multimer: Whether multimer model or not.
        enable_ptm: Enable predicted aligned error related modules.
        enable_templates: Enable template related modules.
        fuse_projection_weights: Whether triangular multiplicative layers are fused or not.
    """

    # Multimer model can predict TM score.
    enable_ptm |= is_multimer

    data = np.load(npz_path, allow_pickle=True)
    if "arr_0" in data:
        data = data["arr_0"].flat[0]
        global _NPZ_KEY_PREFIX
        _NPZ_KEY_PREFIX = "deepfold_batch/deepfold/deepfold_iteration/"
        keys = list(data.keys())
        for key in keys:
            for subkey in data[key]:
                data[key + "//" + subkey] = data[key][subkey]
            del data[key]
    #######################
    # Some templates
    #######################

    LinearWeight = lambda l: (Param(l, param_type=ParamType.LinearWeight))
    LinearWeightSwap = lambda l: (Param(l, param_type=ParamType.LinearWeight, swap=True))

    LinearBias = lambda l: (Param(l))
    LinearBiasSwap = lambda l: (Param(l, swap=True))

    LinearWeightMHA = lambda l: (Param(l, param_type=ParamType.LinearWeightMHA))

    LinearBiasMHA = lambda b: (Param(b, param_type=ParamType.LinearBiasMHA))

    LinearWeightOPM = lambda l: (Param(l, param_type=ParamType.LinearWeightOPM))

    LinearParams = lambda l: {
        "weights": LinearWeight(l.weight),
        "bias": LinearBias(l.bias),
    }
    LinearLeftParams = lambda l, index: {
        "weights": LinearWeight(l.weight[:index, :]),
        "bias": LinearBias(l.bias[:index]),
    }
    LinearRightParams = lambda l, index: {
        "weights": LinearWeight(l.weight[index:, :]),
        "bias": LinearBias(l.bias[index:]),
    }
    LinearSwapParams = lambda l, index: {
        "weights": LinearWeightSwap(l.weight),
        "bias": LinearBiasSwap(l.bias),
    }

    LinearMHAParams = lambda l: {
        "weights": LinearWeightMHA(l.weight),
        "bias": LinearBiasMHA(l.bias),
    }

    LinearNoBiasParams = lambda l: {
        "weights": LinearWeight(l.weight),
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

    if fuse_projection_weights:
        TriMulOutParams = lambda tri_mul: {
            "left_norm_input": LayerNormParams(tri_mul.layer_norm_in),
            # "left_projection": LinearLeftParams(tri_mul.linear_ab_p, tri_mul.linear_ab_p.weight.shape[0]//2),
            # "right_projection":  LinearRightParams(tri_mul.linear_ab_p, tri_mul.linear_ab_p.weight.shape[0]//2),
            "projection": LinearParams(tri_mul.linear_ab_p),
            # "left_gate":  LinearLeftParams(tri_mul.linear_ab_g, tri_mul.linear_ab_g.weight.shape[0]//2),
            # "right_gate":  LinearRightParams(tri_mul.linear_ab_g, tri_mul.linear_ab_g.weight.shape[0]//2),
            "gate": LinearParams(tri_mul.linear_ab_g),
            "center_norm": LayerNormParams(tri_mul.layer_norm_out),
            "output_projection": LinearParams(tri_mul.linear_z),
            "gating_linear": LinearParams(tri_mul.linear_g),
        }

        # see commit b88f8da on the Alphafold repo
        # Alphafold swaps the pseudocode's a and b between the incoming/outcoming iterations of
        # triangle multiplication, which is confusing and not reproduced in our implementation.
        TriMulInParams = lambda tri_mul: {
            "left_norm_input": LayerNormParams(tri_mul.layer_norm_in),
            # "left_projection":  LinearRightParams(tri_mul.linear_ab_p, tri_mul.linear_ab_p.weight.shape[0]//2),
            # "right_projection":  LinearLeftParams(tri_mul.linear_ab_p, tri_mul.linear_ab_p.weight.shape[0]//2),
            "projection": LinearSwapParams(tri_mul.linear_ab_p, tri_mul.linear_ab_p.weight.shape[0] // 2),
            "gate": LinearSwapParams(tri_mul.linear_ab_g, tri_mul.linear_ab_g.weight.shape[0] // 2),
            # "left_gate":  LinearRightParams(tri_mul.linear_ab_g, tri_mul.linear_ab_g.weight.shape[0]//2),
            # "right_gate":  LinearLeftParams(tri_mul.linear_ab_g, tri_mul.linear_ab_g.weight.shape[0]//2),
            "center_norm": LayerNormParams(tri_mul.layer_norm_out),
            "output_projection": LinearParams(tri_mul.linear_z),
            "gating_linear": LinearParams(tri_mul.linear_g),
        }

    else:
        TriMulOutParams = lambda tri_mul: {
            "layer_norm_input": LayerNormParams(tri_mul.layer_norm_in),
            "left_projection": LinearLeftParams(tri_mul.linear_ab_p, tri_mul.linear_ab_p.weight.shape[0] // 2),
            "right_projection": LinearRightParams(tri_mul.linear_ab_p, tri_mul.linear_ab_p.weight.shape[0] // 2),
            "left_gate": LinearLeftParams(tri_mul.linear_ab_g, tri_mul.linear_ab_g.weight.shape[0] // 2),
            "right_gate": LinearRightParams(tri_mul.linear_ab_g, tri_mul.linear_ab_g.weight.shape[0] // 2),
            "center_layer_norm": LayerNormParams(tri_mul.layer_norm_out),
            "output_projection": LinearParams(tri_mul.linear_z),
            "gating_linear": LinearParams(tri_mul.linear_g),
        }

        # see commit b88f8da on the Alphafold repo
        # Alphafold swaps the pseudocode's a and b between the incoming/outcoming iterations of
        # triangle multiplication, which is confusing and not reproduced in our implementation.
        TriMulInParams = lambda tri_mul: {
            "layer_norm_input": LayerNormParams(tri_mul.layer_norm_in),
            "left_projection": LinearRightParams(tri_mul.linear_ab_p, tri_mul.linear_ab_p.weight.shape[0] // 2),
            "right_projection": LinearLeftParams(tri_mul.linear_ab_p, tri_mul.linear_ab_p.weight.shape[0] // 2),
            "left_gate": LinearRightParams(tri_mul.linear_ab_g, tri_mul.linear_ab_g.weight.shape[0] // 2),
            "right_gate": LinearLeftParams(tri_mul.linear_ab_g, tri_mul.linear_ab_g.weight.shape[0] // 2),
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
        "query_norm": LayerNormParams(matt.layer_norm_m),
        "attention": AttentionGatedParams(matt.mha),
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
    if is_multimer:
        MultimerIPAParams = lambda ipa: {
            "q_scalar_projection": {"weights": LinearWeightMHA(ipa.linear_q.weight)},
            "k_scalar_projection": {"weights": LinearWeightMHA(ipa.linear_k.weight)},
            "v_scalar_projection": {"weights": LinearWeightMHA(ipa.linear_v.weight)},
            "q_point_projection": {"point_projection": LinearMHAParams(ipa.linear_q_points.linear)},
            "k_point_projection": {"point_projection": LinearMHAParams(ipa.linear_k_points.linear)},
            "v_point_projection": {"point_projection": LinearMHAParams(ipa.linear_v_points.linear)},
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

    def EvoformerBlockParams(b, is_extra_msa=False):
        if is_extra_msa:
            col_att_name = "msa_column_global_attention"
            msa_col_att_params = MSAGlobalAttParams(b.msa_att_col)
        else:
            col_att_name = "msa_column_attention"
            msa_col_att_params = MSAColAttParams(b.msa_att_col)

        d = {
            "msa_row_attention_with_pair_bias": MSAAttPairBiasParams(b.msa_att_row),
            col_att_name: msa_col_att_params,
            "msa_transition": MSATransitionParams(b.msa_transition),
            "outer_product_mean": OuterProductMeanParams(b.outer_product_mean),
            "triangle_multiplication_outgoing": TriMulOutParams(b.pair_core.tri_mul_out),
            "triangle_multiplication_incoming": TriMulInParams(b.pair_core.tri_mul_in),
            "triangle_attention_starting_node": TriAttParams(b.pair_core.tri_att_start),
            "triangle_attention_ending_node": TriAttParams(b.pair_core.tri_att_end),
            "pair_transition": PairTransitionParams(b.pair_core.pair_transition),
        }

        return d

    ExtraMSABlockParams = partial(EvoformerBlockParams, is_extra_msa=True)

    FoldIterationParams = lambda sm: {
        "invariant_point_attention": IPAParams(sm.ipa),
        "attention_layer_norm": LayerNormParams(sm.layer_norm_ipa),
        "transition": LinearParams(sm.transition.linear_1),
        "transition_1": LinearParams(sm.transition.linear_2),
        "transition_2": LinearParams(sm.transition.linear_3),
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

    if is_multimer:
        MultimerFoldIterationParams = lambda sm: {
            "invariant_point_attention": MultimerIPAParams(sm.ipa),
            "attention_layer_norm": LayerNormParams(sm.layer_norm_ipa),
            "transition": LinearParams(sm.transition.linear_1),
            "transition_1": LinearParams(sm.transition.linear_2),
            "transition_2": LinearParams(sm.transition.linear_3),
            "transition_layer_norm": LayerNormParams(sm.transition.layer_norm),
            "quat_rigid": {"rigid": LinearParams(sm.bb_update.linear)},
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

    ############################
    # translations dict overflow
    ############################
    tps_blocks_params = None
    template_pair_ln = None
    template_angle_emb = None
    template_angle_proj = None
    # if model.template_pair_stack is not None:
    if enable_templates:
        tps_blocks = model.template_pair_stack.blocks
        tps_blocks_params = stacked([TemplatePairBlockParams(b) for b in tps_blocks])
        template_pair_ln = LayerNormParams(model.template_pair_stack.layer_norm)
        template_angle_emb = LinearParams(model.template_angle_embedder.linear_1)
        template_angle_proj = LinearParams(model.template_angle_embedder.linear_2)

    ems_blocks = model.extra_msa_stack.blocks
    ems_blocks_params = stacked([ExtraMSABlockParams(b) for b in ems_blocks])

    evo_blocks = model.evoformer_stack.blocks
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
            "template_embedding": {
                "single_template_embedding": {
                    "template_pair_stack": {
                        "__layer_stack_no_state": tps_blocks_params,
                    },
                    "output_layer_norm": template_pair_ln,
                },
                # "attention": AttentionParams(model.template_pointwise_att.mha),
            },
            "extra_msa_activations": LinearParams(model.extra_msa_embedder.linear),
            "extra_msa_stack": ems_blocks_params,
            "template_single_embedding": template_angle_emb,
            "template_projection": template_angle_proj,
            "evoformer_iteration": evo_blocks_params,
            "single_activations": LinearParams(model.evoformer_stack.linear),
        },
        "structure_module": {
            "single_layer_norm": LayerNormParams(model.structure_module.layer_norm_s),
            "initial_projection": LinearParams(model.structure_module.linear_in),
            "pair_layer_norm": LayerNormParams(model.structure_module.layer_norm_z),
            "fold_iteration": (MultimerFoldIterationParams(model.structure_module) if is_multimer else FoldIterationParams(model.structure_module)),
        },
        "predicted_lddt_head": {
            "input_layer_norm": LayerNormParams(model.auxiliary_heads.plddt.layer_norm),
            "act_0": LinearParams(model.auxiliary_heads.plddt.linear_1),
            "act_1": LinearParams(model.auxiliary_heads.plddt.linear_2),
            "logits": LinearParams(model.auxiliary_heads.plddt.linear_3),
        },
        "distogram_head": {
            "half_logits": LinearParams(model.auxiliary_heads.distogram.linear),
        },
        "experimentally_resolved_head": {
            "logits": LinearParams(model.auxiliary_heads.experimentally_resolved.linear),
        },
        "masked_msa_head": {
            "logits": LinearParams(model.auxiliary_heads.masked_msa.linear),
        },
    }

    if not enable_templates:
        evo_dict = translations["evoformer"]
        keys = list(evo_dict.keys())
        for k in keys:
            if "template_" in k:
                evo_dict.pop(k)

    if enable_ptm:
        translations["predicted_aligned_error_head"] = {"logits": LinearParams(model.auxiliary_heads.tm.linear)}

    # fmt: off
    if is_multimer:
        del translations["evoformer"]["pair_activiations"]
        del translations["evoformer"]["template_embedding"]["single_template_embedding"]["template_pair_stack"]  

        translations["predicted_aligned_error_head"] = {"logits": LinearParams(model.auxiliary_heads.tm.linear)}

        translations["evoformer"]["~_relative_encoding"] = {}
        translations["evoformer"]["~_relative_encoding"]["position_activations"] = LinearParams(model.input_embedder.linear_relpos)

        translations["evoformer"]["template_embedding"]["single_template_embedding"]["query_embedding_norm"] = LayerNormParams(model.template_pair_embedder.query_embedding_layer_norm)
        translations["evoformer"]["template_embedding"]["single_template_embedding"]["template_pair_embedding_0"] = LinearParams(model.template_pair_embedder.dgram_linear)
        translations["evoformer"]["template_embedding"]["single_template_embedding"]["template_pair_embedding_1"] = LinearParams(model.template_pair_embedder.pseudo_beta_mask_linear)
        translations["evoformer"]["template_embedding"]["single_template_embedding"]["template_pair_embedding_2"] = LinearParams(model.template_pair_embedder.aatype_linear_1)
        translations["evoformer"]["template_embedding"]["single_template_embedding"]["template_pair_embedding_3"] = LinearParams(model.template_pair_embedder.aatype_linear_2)
        translations["evoformer"]["template_embedding"]["single_template_embedding"]["template_pair_embedding_4"] = LinearParams(model.template_pair_embedder.x_linear)
        translations["evoformer"]["template_embedding"]["single_template_embedding"]["template_pair_embedding_5"] = LinearParams(model.template_pair_embedder.y_linear)
        translations["evoformer"]["template_embedding"]["single_template_embedding"]["template_pair_embedding_6"] = LinearParams(model.template_pair_embedder.z_linear)
        translations["evoformer"]["template_embedding"]["single_template_embedding"]["template_pair_embedding_7"] = LinearParams(model.template_pair_embedder.backbone_mask_linear)
        translations["evoformer"]["template_embedding"]["single_template_embedding"]["template_pair_embedding_8"] = LinearParams(model.template_pair_embedder.query_embedding_linear)
        translations["evoformer"]["template_embedding"]["single_template_embedding"]["template_embedding_iteration"] = tps_blocks_params
        translations["evoformer"]["template_embedding"]["output_linear"] = LinearParams(model.template_projection.linear_t)
    else:
        if enable_templates:
            translations["evoformer"]["template_embedding"]["single_template_embedding"]["embedding2d"] = LinearParams(model.template_pair_embedder.linear)
            translations["evoformer"]["template_embedding"]["attention"] = AttentionParams(model.template_pointwise_attention.mha)
    # fmt: on

    # Flatten keys and insert missing key prefixes
    flat = _process_translations_dict(translations)
    # Sanity check
    keys = list(data.keys())
    flat_keys = list(flat.keys())

    incorrect = [k for k in flat_keys if k not in keys]
    missing = [k for k in keys if k not in flat_keys]
    # assert len(missing) == 0
    # assert(sorted(list(flat.keys())) == sorted(list(data.keys())))
    logger.debug("incorrect keys:", incorrect)  # which with error names
    logger.debug("missing keys:", missing)  # which with
    # Set weights
    assign(flat, data)
