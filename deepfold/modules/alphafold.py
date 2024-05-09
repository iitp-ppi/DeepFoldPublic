import logging
from dataclasses import asdict
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.common.residue_constants as rc
import deepfold.distributed as dist
import deepfold.distributed.model_parallel as mp
import deepfold.modules.inductor as inductor
from deepfold.config import AlphaFoldConfig
from deepfold.modules.auxiliary_heads import AuxiliaryHeads
from deepfold.modules.evoformer_stack import EvoformerStack
from deepfold.modules.extra_msa_embedder import ExtraMSAEmbedder
from deepfold.modules.extra_msa_stack import ExtraMSAStack
from deepfold.modules.input_embedder import InputEmbedder, InputEmbedderMultimer
from deepfold.modules.recycling_embedder import RecyclingEmbedder  # OpenFoldRecyclingEmbedder
from deepfold.modules.structure_module import StructureModule
from deepfold.modules.template_angle_embedder import TemplateAngleEmbedder
from deepfold.modules.template_pair_embedder import TemplatePairEmbedder, TemplatePairEmbedderMultimer
from deepfold.modules.template_pair_stack import TemplatePairStack
from deepfold.modules.template_pointwise_attention import TemplatePointwiseAttention
from deepfold.modules.template_projection import TemplateProjection
from deepfold.utils.tensor_utils import batched_gather, tensor_tree_map

logger = logging.getLogger(__name__)


class AlphaFold(nn.Module):
    """AlphaFold2 module.

    Supplementary '1.4 AlphaFold Inference': Algorithm 2.

    """

    def __init__(self, config: AlphaFoldConfig) -> None:
        super().__init__()

        if not config.is_multimer:
            self.input_embedder = InputEmbedder(
                **asdict(config.input_embedder_config),
            )
        else:
            self.input_embedder = InputEmbedderMultimer(
                **asdict(config.input_embedder_config),
            )

        self.recycling_embedder = RecyclingEmbedder(
            **asdict(config.recycling_embedder_config),
        )
        if config.templates_enabled:
            self.template_angle_embedder = TemplateAngleEmbedder(
                **asdict(config.template_angle_embedder_config),
            )
            self.template_pair_stack = TemplatePairStack(
                **asdict(config.template_pair_stack_config),
            )
            if not config.is_multimer:
                self.template_pair_embedder = TemplatePairEmbedder(
                    **asdict(config.template_pair_embedder_config),
                )
                self.template_pointwise_attention = TemplatePointwiseAttention(
                    **asdict(config.template_pointwise_attention_config),
                )
            else:
                self.template_pair_embedder = TemplatePairEmbedderMultimer(
                    **asdict(config.template_pair_embedder_config),
                )
                self.template_projection = TemplateProjection(
                    **asdict(config.template_projection_config),
                )
        self.extra_msa_embedder = ExtraMSAEmbedder(
            **asdict(config.extra_msa_embedder_config),
        )

        self.extra_msa_stack = ExtraMSAStack(
            **asdict(config.extra_msa_stack_config),
        )
        self.evoformer_stack = EvoformerStack(
            **asdict(config.evoformer_stack_config),
        )

        self.structure_module = StructureModule(
            **asdict(config.structure_module_config),
        )
        self.auxiliary_heads = AuxiliaryHeads(config.auxiliary_heads_config)

        self.config = config

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        recycle_hook: Callable[[int, dict, dict], None] | None = None,
    ) -> Dict[str, torch.Tensor]:
        # Initialize previous recycling embeddings:
        prevs = self._initialize_prevs(batch)

        # Asym id for multimer
        asym_id = None
        if "asym_id" in batch:  # NOTE: Multimer
            asym_id = batch["asym_id"][..., -1].contiguous()

        # Forward iterations with autograd disabled:
        num_recycling_iters = batch["aatype"].shape[-1] - 1
        for recycle_iter in range(num_recycling_iters):
            feats = tensor_tree_map(fn=lambda t: t[..., recycle_iter].contiguous(), tree=batch)
            with torch.no_grad():
                outputs, prevs = self._forward_iteration(
                    feats=feats,
                    prevs=prevs,
                    gradient_checkpointing=False,
                )

                if recycle_hook is not None:  # Inference
                    aux_outputs = self.auxiliary_heads(outputs, asym_id)
                    outputs.update(aux_outputs)
                    recycle_hook(recycle_iter, feats, outputs)

                del outputs
        recycle_iter += 1  # For the last iteration

        # https://github.com/pytorch/pytorch/issues/65766
        if torch.is_autocast_enabled():
            torch.clear_autocast_cache()

        # Final iteration with autograd enabled:
        feats = tensor_tree_map(fn=lambda t: t[..., -1].contiguous(), tree=batch)
        outputs, prevs = self._forward_iteration(
            feats=feats,
            prevs=prevs,
            gradient_checkpointing=(self.training and mp.size() <= 1),
        )
        del prevs

        outputs["msa"] = outputs["msa"].to(dtype=torch.float32)
        outputs["pair"] = outputs["pair"].to(dtype=torch.float32)
        outputs["single"] = outputs["single"].to(dtype=torch.float32)

        # Run auxiliary heads:
        aux_outputs = self.auxiliary_heads(outputs, asym_id)
        outputs.update(aux_outputs)

        if recycle_hook is not None:  # Inference
            recycle_hook(recycle_iter, feats, outputs)

        return outputs

    def _forward_iteration(
        self,
        feats: Dict[str, torch.Tensor],
        prevs: Dict[str, torch.Tensor],
        gradient_checkpointing: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        outputs = {}

        # batch_size = feats["aatype"].shape[0]
        # num_res = feats["aatype"].shape[1]
        num_clust = feats["msa_feat"].shape[1]

        seq_mask = feats["seq_mask"]
        # seq_mask: [batch, N_res]

        pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)  # outer product
        # pair_mask: [batch, N_res, N_res]

        msa_mask = feats["msa_mask"]
        # msa_mask: [batch, N_clust, N_res]

        # Initialize MSA and pair representations:
        if not self.config.is_multimer:
            m, z = self.input_embedder(
                target_feat=feats["target_feat"],
                residue_index=feats["residue_index"],
                msa_feat=feats["msa_feat"],
            )
        else:
            m, z = self.input_embedder(
                target_feat=feats["target_feat"],
                residue_index=feats["residue_index"],
                msa_feat=feats["msa_feat"],
                asym_id=feats["asym_id"],
                entity_id=feats["entity_id"],
                sym_id=feats["sym_id"],
            )
        # m: [batch, N_clust, N_res, c_m]
        # z: [batch, N_res, N_res, c_z]

        # Extract recycled representations:
        m0_prev = prevs.pop("m0_prev")
        z_prev = prevs.pop("z_prev")
        x_prev = prevs.pop("x_prev")

        x_prev = _pseudo_beta(
            aatype=feats["aatype"],
            all_atom_positions=x_prev,
            dtype=z.dtype,
        )

        m, z = self.recycling_embedder(
            m=m,
            z=z,
            m0_prev=m0_prev,
            z_prev=z_prev,
            x_prev=x_prev,
        )

        del m0_prev, z_prev, x_prev

        # Embed templates and merge with MSA/pair representation:
        if self.config.templates_enabled:
            template_feats = {k: t for k, t in feats.items() if k.startswith("template_")}
            template_embeds = self._embed_templates(
                feats=template_feats,
                z=z,
                pair_mask=pair_mask,
                asym_id=feats["asym_id"] if self.config.is_multimer else None,
                gradient_checkpointing=gradient_checkpointing,
                multichain_mask_2d=feats.get("template_multichain_mask_2d", None),
            )
            # multichain_mask_2d: [batch, N_res, N_res, N_templ]

            z = z + template_embeds["template_pair_embedding"]
            # z: [batch, N_res, N_res, c_z]

            if self.config.embed_template_torsion_angles:
                m = torch.cat([m, template_embeds["template_angle_embedding"]], dim=-3)
                # m: [batch, N_seq, N_res, c_m]

                if not self.config.is_multimer:
                    msa_mask = torch.cat(
                        [
                            feats["msa_mask"],
                            feats["template_torsion_angles_mask"][..., 2],
                        ],
                        dim=-2,
                    )
                    # msa_mask: [batch, N_seq, N_res]
                else:
                    msa_mask = torch.cat(
                        [
                            feats["msa_mask"],
                            template_embeds["template_mask"],
                        ],
                        dim=-2,
                    )

            del template_feats, template_embeds

        # num_seq = m.shape[1]

        # Embed extra MSA features and merge with pairwise embeddings:
        if self.config.is_multimer:
            extra_msa_fn = _build_extra_msa_feat_multimer
        else:
            extra_msa_fn = _build_extra_msa_feat
        # N_extra_seq = feats["extra_msa"].shape[1]
        a = self.extra_msa_embedder(extra_msa_fn(feats))
        # a: [batch, N_extra_seq, N_res, c_e]
        z = self.extra_msa_stack(
            m=a,
            z=z,
            msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=m.dtype),
            gradient_checkpointing=gradient_checkpointing,
        )
        # z: [batch, N_res, N_res, c_z]
        del a

        # Evoformer forward pass:
        m, z, s = self.evoformer_stack(
            m=m,
            z=z,
            msa_mask=msa_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            gradient_checkpointing=gradient_checkpointing,
        )
        # m: [batch, N_seq, N_res, c_m]
        # z: [batch, N_res, N_res, c_z]
        # s: [batch, N_res, c_s]
        outputs["msa"] = m[:, :num_clust]
        outputs["pair"] = z
        outputs["single"] = s

        # Predict 3D structure:
        sm_outputs = self.structure_module(
            s=outputs["single"].to(dtype=torch.float32),
            z=outputs["pair"].to(dtype=torch.float32),
            mask=feats["seq_mask"].to(dtype=s.dtype),
            aatype=feats["aatype"],
        )

        outputs.update(sm_outputs)
        outputs["final_atom_positions"] = _atom14_to_atom37(
            atom14_positions=outputs["sm_positions"][:, -1],
            residx_atom37_to_atom14=feats["residx_atom37_to_atom14"],
            atom37_atom_exists=feats["atom37_atom_exists"],
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"].to(dtype=outputs["final_atom_positions"].dtype)
        outputs["final_affine_tensor"] = outputs["sm_frames"][:, -1]

        # Save embeddings for next recycling iteration:
        prevs = {}
        prevs["m0_prev"] = m[:, 0]
        prevs["z_prev"] = outputs["pair"]
        prevs["x_prev"] = outputs["final_atom_positions"]

        return outputs, prevs

    def _initialize_prevs(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        prevs = {}
        batch_size = batch["aatype"].shape[0]
        num_res = batch["aatype"].shape[1]
        c_m = self.input_embedder.c_m
        c_z = self.input_embedder.c_z
        device = batch["msa_feat"].device
        dtype = batch["msa_feat"].dtype
        prevs["m0_prev"] = torch.zeros(
            size=[batch_size, num_res, c_m],
            device=device,
            dtype=dtype,
        )
        prevs["z_prev"] = torch.zeros(
            size=[batch_size, num_res, num_res, c_z],
            device=device,
            dtype=dtype,
        )
        prevs["x_prev"] = torch.zeros(
            size=[batch_size, num_res, rc.atom_type_num, 3],
            device=device,
            dtype=torch.float32,
        )
        return prevs

    def _embed_templates(
        self,
        feats: Dict[str, torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        gradient_checkpointing: bool,
        asym_id: Optional[torch.Tensor] = None,
        multichain_mask_2d: Optional[torch.Tensor] = None,  # [..., N_res, N_res, N_templ]
    ) -> Dict[str, torch.Tensor]:
        # Embed the templates one at a time:
        pair_embeds = []
        num_templ = feats["template_aatype"].shape[1]
        for i in range(num_templ):
            single_template_feats = tensor_tree_map(fn=lambda t: t[:, i], tree=feats)
            if multichain_mask_2d is not None:
                single_multichain_mask_2d = multichain_mask_2d[..., i]
            else:
                single_multichain_mask_2d = None

            if not self.config.is_multimer:
                t = self.template_pair_embedder.build_template_pair_feat(
                    feats=single_template_feats,
                    min_bin=self.config.template_pair_feat_distogram_min_bin,
                    max_bin=self.config.template_pair_feat_distogram_max_bin,
                    num_bins=self.config.template_pair_feat_distogram_num_bins,
                    use_unit_vector=self.config.template_pair_feat_use_unit_vector,
                    inf=self.config.template_pair_feat_inf,
                    eps=self.config.template_pair_feat_eps,
                    dtype=z.dtype,
                )

                t = self.template_pair_embedder(t)
                # t: [batch, N_res, N_res, c_t]
            else:
                if single_multichain_mask_2d is None:
                    single_multichain_mask_2d = asym_id[..., :, None] == asym_id[..., None, :]
                # single_multichain_mask_2d: [batch, N_res, N_res]

                t = self.template_pair_embedder.build_template_pair_feat(
                    feats=single_template_feats,
                    min_bin=self.config.template_pair_feat_distogram_min_bin,
                    max_bin=self.config.template_pair_feat_distogram_max_bin,
                    num_bins=self.config.template_pair_feat_distogram_num_bins,
                    inf=self.config.template_pair_feat_inf,
                    eps=self.config.template_pair_feat_eps,
                    dtype=z.dtype,
                )

                t = self.template_pair_embedder(
                    query_embedding=z,
                    multichain_mask_2d=single_multichain_mask_2d,
                    **t,
                )
                # t: [batch, N_res, N_res, c_t]

            pair_embeds.append(t)
            del t

        t = torch.stack(pair_embeds, dim=1)
        # t: [batch, N_templ, N_res, N_res, c_t]
        del pair_embeds

        t = self.template_pair_stack(
            t=t,
            mask=pair_mask.to(dtype=z.dtype),
            gradient_checkpointing=gradient_checkpointing,
        )
        # t: [batch, N_templ, N_res, N_res, c_t]

        if self.config.is_multimer:
            t = self.template_projection(t=t)
        else:
            t = self.template_pointwise_attention(
                t=t,
                z=z,
                template_mask=feats["template_mask"].to(dtype=z.dtype),
            )
            t = _apply_template_mask(t=t, template_mask=feats["template_mask"])
        # t: [batch, N_res, N_res, c_z]

        template_embeds = {}
        template_embeds["template_pair_embedding"] = t

        if self.config.embed_template_torsion_angles:
            if self.config.is_multimer:
                template_angle_feat, template_mask = _build_template_angle_feat_multimer(feats)
                template_embeds["template_mask"] = template_mask
            else:
                template_angle_feat = _build_template_angle_feat(feats)
            a = self.template_angle_embedder(template_angle_feat)
            # a: [batch, N_templ, N_res, c_m]
            template_embeds["template_angle_embedding"] = a

        return template_embeds

    def register_dap_gradient_scaling_hooks(self, dap_size: int) -> None:
        num_registered_hooks = {
            "evoformer_stack": 0,
            "extra_msa_stack": 0,
            "template_pair_stack": 0,
        }

        evoformer_stack = self.evoformer_stack

        for name, param in evoformer_stack.named_parameters():
            if name.startswith("blocks."):
                param.register_hook(lambda grad: grad * dap_size)
                num_registered_hooks["evoformer_stack"] += 1

        for name, param in self.extra_msa_stack.named_parameters():
            if name.startswith("blocks."):
                param.register_hook(lambda grad: grad * dap_size)
                num_registered_hooks["extra_msa_stack"] += 1

        if hasattr(self, "template_pair_stack"):
            for name, param in self.template_pair_stack.named_parameters():
                if name.startswith("blocks."):
                    param.register_hook(lambda grad: grad * dap_size)
                    num_registered_hooks["template_pair_stack"] += 1

        if dist.is_main_process():
            logger.info("register_dap_gradient_scaling_hooks: " f"num_registered_hooks={num_registered_hooks}")


def _pseudo_beta_eager(
    aatype: torch.Tensor,
    all_atom_positions: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    is_gly = torch.eq(aatype, rc.restype_order["G"])
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    pseudo_beta = torch.where(
        torch.tile(is_gly.unsqueeze(-1), [1] * is_gly.ndim + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )
    return pseudo_beta.to(dtype=dtype)


_pseudo_beta_jit = torch.compile(_pseudo_beta_eager)


def _pseudo_beta(
    aatype: torch.Tensor,
    all_atom_positions: torch.Tensor,
    dtype,
) -> torch.Tensor:
    if inductor.is_enabled():
        pseudo_beta_fn = _pseudo_beta_jit
    else:
        pseudo_beta_fn = _pseudo_beta_eager
    return pseudo_beta_fn(
        aatype,
        all_atom_positions,
        dtype,
    )


def _apply_template_mask_eager(t: torch.Tensor, template_mask: torch.Tensor) -> torch.Tensor:
    t_mask = (torch.sum(template_mask, dim=1) > 0).to(dtype=t.dtype)
    t_mask = t_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    t = t * t_mask
    return t


_apply_template_mask_jit = torch.compile(_apply_template_mask_eager)


def _apply_template_mask(t: torch.Tensor, template_mask: torch.Tensor) -> torch.Tensor:
    if inductor.is_enabled():
        apply_template_mask_fn = _apply_template_mask_jit
    else:
        apply_template_mask_fn = _apply_template_mask_eager
    return apply_template_mask_fn(t, template_mask)


def _build_extra_msa_feat_eager(
    extra_msa: torch.Tensor,
    extra_has_deletion: torch.Tensor,
    extra_deletion_value: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    msa_1hot = F.one_hot(input=extra_msa, num_classes=num_classes)
    msa_feat = [
        msa_1hot,
        extra_has_deletion.unsqueeze(-1),
        extra_deletion_value.unsqueeze(-1),
    ]
    return torch.cat(msa_feat, dim=-1)


_build_extra_msa_feat_jit = torch.compile(_build_extra_msa_feat_eager)


def _build_extra_msa_feat(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    if inductor.is_enabled():
        build_extra_msa_feat_fn = _build_extra_msa_feat_jit
    else:
        build_extra_msa_feat_fn = _build_extra_msa_feat_eager
    return build_extra_msa_feat_fn(
        extra_msa=feats["extra_msa"],
        extra_has_deletion=feats["extra_has_deletion"],
        extra_deletion_value=feats["extra_deletion_value"],
        num_classes=23,
    )


def _build_extra_msa_feat_multimer_eager(
    extra_msa: torch.Tensor,
    extra_deletion_matrix: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    msa_1hot = F.one_hot(extra_msa, num_classes=num_classes)
    has_deletion = torch.clamp(extra_deletion_matrix, min=0.0, max=1.0)[..., None]
    deletion_value = (torch.atan(extra_deletion_matrix / 3.0) * (2.0 / torch.pi))[..., None]
    return torch.cat([msa_1hot, has_deletion, deletion_value], dim=-1)


_build_extra_msa_feat_multimer_jit = torch.compile(_build_extra_msa_feat_multimer_eager)


def _build_extra_msa_feat_multimer(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
    if inductor.is_enabled():
        build_extra_msa_feat_fn = _build_extra_msa_feat_multimer_eager
    else:
        build_extra_msa_feat_fn = _build_extra_msa_feat_multimer_jit
    return build_extra_msa_feat_fn(
        extra_msa=feats["extra_msa"],
        extra_deletion_matrix=feats["extra_deletion_matrix"],
        num_classes=23,
    )


def _build_template_angle_feat(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    template_aatype = feats["template_aatype"]
    torsion_angles_sin_cos = feats["template_torsion_angles_sin_cos"]
    alt_torsion_angles_sin_cos = feats["template_alt_torsion_angles_sin_cos"]
    torsion_angles_mask = feats["template_torsion_angles_mask"]
    template_angle_feat = torch.cat(
        [
            F.one_hot(input=template_aatype, num_classes=22),
            torsion_angles_sin_cos.reshape(*torsion_angles_sin_cos.shape[:-2], 14),
            alt_torsion_angles_sin_cos.reshape(*alt_torsion_angles_sin_cos.shape[:-2], 14),
            torsion_angles_mask,
        ],
        dim=-1,
    )
    return template_angle_feat


def _build_template_angle_feat_multimer(feats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    template_aatype = feats["template_aatype"]
    template_all_atom_positions = feats["template_all_atom_positions"]
    template_all_atom_mask = feats["template_all_atom_mask"]
    dtype = template_all_atom_positions.dtype

    template_chi_angles, template_chi_mask = _compute_chi_angles(
        positions=template_all_atom_positions,
        mask=template_all_atom_mask,
        aatype=template_aatype,
    )

    template_angle_feat = torch.cat(
        [
            F.one_hot(input=template_aatype, num_classes=22),
            torch.sin(template_chi_angles) * template_chi_mask,
            torch.cos(template_chi_angles) * template_chi_mask,
            template_chi_mask,
        ],
        dim=-1,
    ).to(dtype)

    # NOTE: Multimer model gets `template_mask` from the angle features.
    template_mask = template_chi_mask[..., 0].to(dtype=dtype)

    return template_angle_feat, template_mask


def _atom14_to_atom37(
    atom14_positions: torch.Tensor,
    residx_atom37_to_atom14: torch.Tensor,
    atom37_atom_exists: torch.Tensor,
) -> torch.Tensor:
    # atom14_positions: [batch, N_res, 14, 3]
    # residx_atom37_to_atom14: [batch, N_res, 37]
    # atom37_atom_exists: [batch, N_res, 37]

    indices = residx_atom37_to_atom14.unsqueeze(-1)
    # indices: [batch, N_res, 37, 1]
    indices = indices.expand(-1, -1, -1, 3)
    # indices: [batch, N_res, 37, 3]

    atom37_positions = torch.gather(atom14_positions, 2, indices)
    # atom37_positions: [batch, N_res, 37, 3]

    atom37_mask = atom37_atom_exists.unsqueeze(-1)
    # atom37_mask: [batch, N_res, 37, 1]

    atom37_positions = atom37_positions * atom37_mask
    # atom37_positions: [batch, N_res, 37, 3]

    return atom37_positions


def _compute_chi_angles(
    positions: torch.Tensor,
    mask: torch.Tensor,
    aatype: torch.Tensor,
    chi_atom_indices: Optional[torch.Tensor] = None,
    chi_angles_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the chi angles given all atom positions and the amino acid type.

    Args:
        positions: [..., 37, 3] Atom positions in atom37 format.
        atom_mask: [..., 37] Atom mask.
        aatype: [...] Amino acid type integer code.

    Returns:
        chi_angles: [batch, N_res, 4].
        chi_mask: [batch, N_res, 4].
    """
    assert positions.shape[-2] == rc.atom_type_num
    assert mask.shape[-1] == rc.atom_type_num
    num_batch_dims = aatype.ndim

    if chi_atom_indices is None:
        chi_atom_indices = positions.new_tensor(rc.CHI_ATOM_INDICES, dtype=torch.int64)
        # chi_atom_indices: [restype=21, chis=4, atoms=4]

    # Remove gaps
    aatype = torch.clamp(aatype, max=20)

    # Select atoms to compute chis.
    atom_indices = chi_atom_indices[..., aatype, :, :]
    # atom_indices: [batch, N_res, chis=4, atoms=4]

    x, y, z = torch.unbind(positions, dim=-1)
    x = batched_gather(x, atom_indices, -1, num_batch_dims).unsqueeze(-1)
    y = batched_gather(y, atom_indices, -1, num_batch_dims).unsqueeze(-1)
    z = batched_gather(z, atom_indices, -1, num_batch_dims).unsqueeze(-1)
    xyz = torch.cat([x, y, z], dim=-1)
    a, b, c, d = torch.unbind(xyz, dim=-2)

    chi_angles = _dihedral_angle(a, b, c, d)

    if chi_angles_mask is None:
        chi_angles_mask = list(rc.chi_angles_mask)
        chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])  # UNK
        chi_angles_mask = mask.new_tensor(chi_angles_mask)

    chi_mask = chi_angles_mask[aatype, :]
    # chi_mask[batch, N_res, chi=4]

    chi_angle_atoms_mask = batched_gather(mask, atom_indices, -1, num_batch_dims)
    chi_angle_atoms_mask = torch.prod(chi_angle_atoms_mask, -1, dtype=chi_angle_atoms_mask.dtype)
    chi_mask = chi_mask * chi_angle_atoms_mask

    return chi_angles, chi_mask


def _dihedral_angle(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Computes torsion angle for a quadruple of points.

    For points (a, b, c, d), this is the angle between the planes defined by
    points (a, b, c) and (b, c, d). It is also known as the dihedral angle.

    """
    v1 = a - b
    v2 = b - c
    v3 = d - c

    c1 = v1.cross(v2, dim=-1)
    c2 = v3.cross(v2, dim=-1)
    c3 = c2.cross(c1, dim=-1)

    first = torch.einsum("...i,...i", c3, v2)
    v2_mag = v2.norm(dim=-1)
    second = v2_mag * torch.einsum("...i,...i", c1, c2)

    return torch.atan2(first, second)
