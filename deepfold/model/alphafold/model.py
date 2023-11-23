# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited

from typing import List, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

import deepfold.common.residue_constants as rc
from deepfold.distributed.legacy import get_tensor_model_parallel_world_size
from deepfold.model.alphafold.dist_layers import GatherOutputs, ScatterFeatures
from deepfold.model.alphafold.feats import (
    atom14_to_atom37,
    build_extra_msa_feat,
    build_template_angle_feat,
    build_template_pair_feat,
    pseudo_beta_fn,
)
from deepfold.model.alphafold.loss import compute_tm
from deepfold.model.alphafold.nn.embedders import (
    ExtraMSAEmbedder,
    ParallelInputEmbedder,
    ParallelRecyclingEmbedder,
    TemplateAngleEmbedder,
    TemplatePairEmbedder,
)
from deepfold.model.alphafold.nn.evoformer import EvoformerStack, ExtraMSAStack
from deepfold.model.alphafold.nn.heads import AuxiliaryHeads
from deepfold.model.alphafold.nn.structure_module import StructureModule
from deepfold.model.alphafold.nn.template import TemplatePairStack, TemplatePointwiseAttention
from deepfold.model.alphafold.pipeline.types import TensorDict
from deepfold.utils.tensor_utils import tensor_tree_map


class AlphaFold(nn.Module):
    """
    Implements Algorithm 2.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.globals = config.globals
        self.config = config.model
        self.template_cfg = self.config.template
        self.extra_msa_cfg = self.config.extra_msa
        # Main trunk and structure module

        self.scatter_features = ScatterFeatures()

        self.input_embedder = ParallelInputEmbedder(
            **self.config["input_embedder"],
        )
        self.recycling_embedder = ParallelRecyclingEmbedder(
            **self.config["recycling_embedder"],
        )

        if self.template_cfg.enabled:
            self.template_angle_embedder = TemplateAngleEmbedder(
                **self.template_cfg["template_angle_embedder"],
            )
            self.template_pair_embedder = TemplatePairEmbedder(
                **self.template_cfg["template_pair_embedder"],
            )
            self.template_pair_stack = TemplatePairStack(
                **self.template_cfg["template_pair_stack"],
            )
            self.template_pointwise_att = TemplatePointwiseAttention(
                **self.template_cfg["template_pointwise_attention"]
            )

        if self.extra_msa_cfg.enabled:
            self.extra_msa_embedder = ExtraMSAEmbedder(
                **self.extra_msa_cfg["extra_msa_embedder"],
            )
            self.extra_msa_stack = ExtraMSAStack(
                **self.extra_msa_cfg["extra_msa_stack"],
            )

        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )
        self.structure_module = StructureModule(
            **self.config["structure_module"],
        )
        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )

        self.gather_outputs = GatherOutputs(self.config)

    def embed_templates(
        self,
        batch: TensorDict,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        templ_dim: int,
    ) -> TensorDict:
        if self.template_cfg.average_templates:
            raise NotImplementedError("Average templates are not implemented yet")

        pair_embeds = []
        n_templ = batch["template_aatype"].shape[templ_dim]

        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, templ_dim, idx).squeeze(templ_dim),
                batch,
            )

            # [*, N, N', C_t]
            t = build_template_pair_feat(
                single_template_feats,
                use_unit_vector=self.template_cfg.use_unit_vector,
                inf=self.template_cfg.inf,
                eps=self.template_cfg.eps,
                **self.template_cfg.distogram,
            ).to(z.dtype)
            t = self.template_pair_embedder(t)

            pair_embeds.append(t)

            del t

        t_pair = torch.stack(pair_embeds, dim=templ_dim)

        del pair_embeds

        # [*, S_t, N', N, C_z]
        t = self.template_pair_stack(
            t_pair,
            pair_mask.unsqueeze(-3).to(dtype=z.dtype),
            chunk_size=self.template_cfg.template_pair_stack.chunk_size,
        )
        del t_pair

        # [*, N', N, C_z]
        t = self.template_pointwise_att(
            t,
            z,
            template_mask=batch["template_mask"].to(dtype=z.dtype),
        )

        t_mask = torch.sum(batch["template_mask"], dim=-1) > 0
        t_mask = t_mask.reshape(*t_mask.shape, *([1] * (len(t.shape) - len(t_mask.shape))))

        t = t * t_mask

        ret = {}

        ret.update({"template_pair_embedding": t})

        del t

        if self.template_cfg.embed_angles:
            template_angle_feat = build_template_angle_feat(batch)

            # [*, S_t, N', C_m]
            a = self.template_angle_embedder(template_angle_feat)

            ret["template_angle_embedding"] = a

        return ret

    def iteration(
        self,
        feats: TensorDict,
        prevs: List[torch.Tensor],
    ) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Output
        outputs = {}

        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        batch_dims = feats["target_feat"].shape[:-2]
        num_batch_dims = len(batch_dims)
        n_row = feats["target_feat"].shape[-2]  # N'
        n_col = n_row * get_tensor_model_parallel_world_size()  # N
        n_seq = feats["msa_feat"].shape[-3]  # S_c

        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., :, None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        # m: [*, S_c, N', C_m]
        # z: [*, N', N, C_z]
        m, z = self.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
        )

        m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)])

        if None in [m_1_prev, z_prev, x_prev]:
            # [*, N', C_m]
            m_1_prev = m.new_zeros((*batch_dims, n_row, self.config.input_embedder.c_m), requires_grad=False)

            # [*, N', N, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n_row, n_col, self.config.input_embedder.c_z),
                requires_grad=False,
            )

            # [*, N', 3]
            x_prev = z.new_zeros(
                (*batch_dims, n_row, rc.atom_type_num, 3),
                requires_grad=False,
            )

        x_prev = pseudo_beta_fn(feats["aatype"], x_prev, None).to(dtype=z.dtype)

        # m_1_prev_emb: [*, N', C_m]
        # z_prev_emb: [*, N', N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(m_1_prev, z_prev, x_prev)

        # [*, S_c, N', C_m]
        m[..., 0, :, :] += m_1_prev_emb

        # [*, N', N, C_z]
        z = z + z_prev_emb

        del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb

        # Embed the templates + merge with MSA/pair embeddings
        if self.config.template.enabled:
            template_feats = {k: v for k, v in feats.items() if k.startswith("template_")}
            template_embeds = self.embed_templates(template_feats, z, pair_mask.to(dtype=z.dtype), num_batch_dims)
            # [*, N', N, C_z]
            z = z + template_embeds.pop("template_pair_embedding")

        if self.config.template.embed_angles and "template_angle_embedding" in template_embeds:
            # [*, S = S_c + S_t, N', C_m]
            m = torch.cat([m, template_embeds["template_angle_embedding"]], dim=-3)

            # [*, S, N']
            torsion_angles_mask = feats["template_torsion_angles_mask"]
            msa_mask = torch.cat([feats["msa_mask"], torsion_angles_mask[..., 2]], dim=-2)

        # Embed extra MSA features + merge with pairwise embeddings
        if self.config.extra_msa.enabled:
            # [*, S_e, N', C_e]
            a = self.extra_msa_embedder(build_extra_msa_feat(feats))

            # [*, N', N, C_z]
            z = self.extra_msa_stack(
                a,
                z,
                msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                chunk_size=self.extra_msa_cfg.extra_msa_stack.chunk_size,
                pair_mask=pair_mask.to(dtype=m.dtype),
            )

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N', C_m]
        # z: [*, N', N, C_z]
        # s: [*, N', C_s]
        m, z, s = self.evoformer(
            m,
            z,
            msa_mask=msa_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            chunk_size=self.config.evoformer_stack.chunk_size,
        )

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        del z

        # Predict 3D structure
        outputs["sm"] = self.structure_module(outputs, feats["aatype"], mask=feats["seq_mask"].to(dtype=s.dtype))
        outputs["final_atom_positions"] = atom14_to_atom37(outputs["sm"]["positions"][-1], feats)
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        # Save embeddings for use during the next recycling iteration

        # [*, N', C_m]
        m_1_prev = m[..., 0, :, :]

        # [*, N', N, C_z]
        z_prev = outputs["pair"]

        # [*, N', 3]
        x_prev = outputs["final_atom_positions"]

        return outputs, m_1_prev, z_prev, x_prev

    def forward(self, batch: TensorDict) -> TensorDict:
        """
        Args:
            batch: TensorDict
                Features (without the recyling dimension):
                    - aatype [*, N']
                    - target_feat [*, N', tf_dim]
                    - residue_index [*, N']
                    - msa_feat [*, S_m, N', msa_dim]
                    - seq_mask [*, N]
                    - msa_mask [*, S_m, N, msa_dim]
                    - pair_mask [*, N, N]
                    - extra_msa [*, S_e, N']
                    - extra_msa_mask [*, S_e, N]
                    - template_mask [*, N_t]
                    - template_aatype [*, N_t, N']
                    - template_all_atom_positions [*, N_t, N', 37, 3]
                    - template_all_atom_mask [*, N_t, N, 37]
                    - template_pseudo_beta [*, N_t, N', 3]
                    - template_pseudo_beta_mask [*, N_t, N]
        Returns:
            TensorDict
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]

        is_grad_enabled = torch.is_grad_enabled()

        # Main recycling loop
        num_iters = batch["aatype"].shape[-1]
        for cycle_idx in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_idx]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable gradient when training and it's the final recycling layer
            is_final_iter = cycle_idx == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Scatter features
                feats = self.scatter_features(feats)

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev = self.iteration(feats, prevs)

                if not is_final_iter:
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev

        # Run auxiliary heads
        outputs.update(self.aux_heads(outputs))

        # Gather outputs
        outputs = self.gather_outputs(outputs)

        # Calculate pTM score
        if self.config.heads.tm.enabled:
            outputs["predicted_tm_score"] = compute_tm(
                logits=outputs["tm_logits"],
                residue_weights=feats["seq_mask"],
                **self.config.heads.tm,
            )

        return outputs
