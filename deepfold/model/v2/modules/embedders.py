# Copyright 2024 DeepFold Team


"""Embedder layers."""


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from deepfold.model.v2.modules.configuration import (
    ExtraMsaEmbedderConfig,
    InputEmbedderConfig,
    RecyclingEmbedderConfig,
    TemplateAngleEmbedderConfig,
    TemplatePairEmbedderConfig,
)
from deepfold.model.v2.modules.primitives import LayerNorm, Linear, residual
from deepfold.utils.tensor_utils import one_hot


class InputEmbedder(nn.Module):
    def __input__(self, config: InputEmbedderConfig) -> None:
        super().__init__()

        self.tf_dim = config.target_feature_dim
        self.msa_dim = config.msa_feature_dim

        self.d_pair = config.pair_representation_dim
        self.d_msa = config.msa_feature_dim

        # Relative positional embedding
        self.max_relative_idx = config.max_relative_idx
        self.use_chain_relative = config.use_chain_relative
        self.max_relative_chain = config.max_relative_chain

        self.num_bins = 2 * self.max_relative_idx + 1
        if self.use_chain_relative:
            self.num_bins += 1  # != asym_id
            self.num_bins += 1  # same_entity
            self.num_bins += 2 * self.max_relative_chain + 2  # sym_id (rel_chain)

        self.linear_relpos = Linear(self.num_bins, self.d_pair)

    def _relpos_indices(
        self,
        res_id: torch.Tensor,
        sym_id: Optional[torch.Tensor] = None,
        asym_id: Optional[torch.Tensor] = None,
        entity_id: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        max_rel_res = self.relpos_k
        rp = res_id[..., None] - res_id[..., None, :]
        rp = rp.clip(-max_rel_res, max_rel_res) + max_rel_res
        if not self.use_chain_relative:
            return rp
        else:
            asym_id_same = asym_id[..., :, None] == asym_id[..., None, :]
            rp[~asym_id_same] = 2 * max_rel_res + 1
            entity_id_same = entity_id[..., :, None] == entity_id[..., None, :]
            rp_entity_id = entity_id_same.type(rp.dtype)[..., None]

            rel_sym_id = sym_id[..., :, None] - sym_id[..., None, :]

            max_rel_chain = self.max_relative_chain

            clipped_rel_chain = torch.clamp(rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain)

            clipped_rel_chain[~entity_id_same] = 2 * max_rel_chain + 1
            return rp, rp_entity_id, clipped_rel_chain

    def relpos_emb(
        self,
        res_id: torch.Tensor,
        sym_id: Optional[torch.Tensor] = None,
        asym_id: Optional[torch.Tensor] = None,
        entity_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dtype = self.linear_relpos.weight.dtype
        if not self.use_chain_relative:
            rp = self._relpos_indices(res_id=res_id)
            return self.linear_relpos(one_hot(rp, num_classes=self.num_bins, dtype=dtype))
        else:
            rp, rp_entity_id, rp_rel_chain = self._relpos_indices(
                res_id=res_id, sym_id=sym_id, asym_id=asym_id, entity_id=entity_id
            )
            rp = one_hot(rp, num_classes=(2 * self.relpos_k + 2), dtype=dtype)
            rp_entity_id = rp_entity_id.type(dtype)
            rp_rel_chain = one_hot(rp_rel_chain, num_classes=(2 * self.max_relative_chain + 2), dtype=dtype)
            return self.linear_relpos(torch.cat([rp, rp_entity_id, rp_rel_chain], dim=-1))

    def forward(
        self,
        tf: torch.Tensor,
        msa: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [*, N_res, d_pair]
        if self.tf_dim == 21:
            # multimer use 21 target dim
            tf = tf[..., 1:]
        # convert type if necessary
        tf = tf.type(self.linear_tf_z_i.weight.dtype)
        msa = msa.type(self.linear_tf_z_i.weight.dtype)
        n_clust = msa.shape[-3]

        msa_emb = self.linear_msa_m(msa)
        # target_feat (aatype) into msa representation
        tf_m = (
            self.linear_tf_m(tf).unsqueeze(-3).expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))
        )  # expand -3 dim
        msa_emb += tf_m

        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)
        pair_emb = tf_emb_i[..., None, :] + tf_emb_j[..., None, :, :]

        return msa_emb, pair_emb


class RecyclingEmbedder(nn.Moudle):

    def __init__(self, config: RecyclingEmbedderConfig) -> None:
        super().__init__()

        self.d_msa = config.msa_representation_dim
        self.d_pair = config.pair_representation_dim
        self.min_bin = config.min_bin
        self.max_bin = config.max_bin
        self.num_bins = config.num_bins
        self.inf = config.inf

        self.linear = Linear(self.num_bins, self.d_pair)
        self.layer_norm_m = LayerNorm(self.d_msa)
        self.layer_norm_z = LayerNorm(self.d_pair)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        m_update = self.layer_norm_m(m)
        z_update = self.layer_norm_z(z)

        return m_update, z_update

    def recycle_pos(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.num_bins,
            dtype=torch.float if self.training else x.dtype,
            device=x.device,
            requires_grad=False,
        )
        self.squared_bin = bins**2
        upper = torch.cat([self.squared_bin[1:], self.squared_bin.new_tensor([self.inf])], dim=-1)

        if self.training:
            x = x.float()

        d = torch.sum((x[..., :, None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdim=True)
        d = ((d > self.squared_bin) * (d < upper)).type(self.linear.weight.dtype)
        d = self.linear(d)

        return d


class TemplateAngleEmbedder(nn.Module):

    def __init__(self, config: TemplateAngleEmbedderConfig) -> None:
        super().__init__()

        self.d_in = config.template_angle_feature_dim
        self.d_out = config.msa_representation_dim

        self.linear_1 = Linear(self.d_in, self.d_out, init="relu")
        self.act = nn.Relu()
        self.linear_2 = Linear(self.d_out, self.d_out, init="relu")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.linear_1(x.type(self.linear_1.weight.dtype))
        x = self.act(x)
        x = self.linear_2(x)

        return x


class TemplatePairEmbedder(nn.Module):
    def __init__(self, config: TemplatePairEmbedderConfig) -> None:
        self.d_out = config.template_representation_dim
        self.d_in = config.template_feature_dims
        self.v2_feature = config.v2_feature
        self.d_pair = config.pair_representation_dim

        if self.v2_feature:
            self.z_layer_norm = LayerNorm(self.d_out)
            self.z_linear = Linear(self.d_pair, self.d_out, init="relu")
        self.linear = nn.ModuleList()
        for d_in in self.d_in:
            self.linear.append(Linear(d_in, self.d_out, init="relu"))

    def forward(self, x: List[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        if not self.v2_feature:  # AF2
            t = self.linear(x.type(self.linear[0].weight.dtype))
        else:  # AFM
            dtype = self.z_linear.weight.dtype
            t = self.linear[0](x[0].type(dtype))
            for i, s in enumerate(x[1:], start=1):
                t = residual(t, self.linear[i](s.type(dtype)), self.training)
            t = residual(t, self.z_linear(self.z_layer_norm(z)), self.training)

        return t


class ExtraMsaEmbedder(nn.Module):
    def __init__(self, config: ExtraMsaEmbedderConfig) -> None:
        super().__init__()

        self.d_in = config.extra_msa_feature_dim
        self.d_out = config.extra_msa_representation_dim
        self.linear = Linear(self.d_in, self.d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.type(self.linear.weight.dtype))
