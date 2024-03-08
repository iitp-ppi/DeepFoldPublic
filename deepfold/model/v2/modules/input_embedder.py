from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.model.v2.modules.inductor as inductor
from deepfold.model.v2.modules.linear import Linear


class InputEmbedder(nn.Module):
    """Input Embedder module.

    Supplementary '1.5 Input embeddings'.

    Args:
        tf_dim: Input `target_feat` dimension (channels).
        msa_dim: Input `msa_feat` dimension (channels).
        c_z: Output pair representation dimension (channels).
        c_m: Output MSA representation dimension (channels).
        relpos_k: Relative position clip distance.

    """

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        relpos_k: int,
    ) -> None:
        super().__init__()
        self.tf_dim = tf_dim
        self.msa_dim = msa_dim
        self.c_z = c_z
        self.c_m = c_m
        self.relpos_k = relpos_k
        self.num_bins = 2 * relpos_k + 1
        self.linear_tf_z_i = Linear(tf_dim, c_z, bias=True, init="default")
        self.linear_tf_z_j = Linear(tf_dim, c_z, bias=True, init="default")
        self.linear_tf_m = Linear(tf_dim, c_m, bias=True, init="default")
        self.linear_msa_m = Linear(msa_dim, c_m, bias=True, init="default")
        self.linear_relpos = Linear(self.num_bins, c_z, bias=True, init="default")

    def forward(
        self,
        target_feat: torch.Tensor,
        residue_index: torch.Tensor,
        msa_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input Embedder forward pass.

        Supplementary '1.5 Input embeddings': Algorithm 3.

        Args:
            target_feat: [batch, N_res, tf_dim]
            residue_index: [batch, N_res]
            msa_feat: [batch, N_clust, N_res, msa_dim]

        Returns:
            msa_emb: [batch, N_clust, N_res, c_m]
            pair_emb: [batch, N_res, N_res, c_z]

        """
        pair_emb = self._forward_pair_embedding(target_feat, residue_index)
        # pair_emb: [batch, N_res, N_res, c_z]

        msa_emb = self._forward_msa_embedding(msa_feat, target_feat)
        # msa_emb: [batch, N_clust, N_res, c_m]

        return msa_emb, pair_emb

    def _forward_pair_embedding(
        self,
        target_feat: torch.Tensor,
        residue_index: torch.Tensor,
    ) -> torch.Tensor:
        if inductor.is_enabled():
            forward_pair_embedding_fn = _forward_pair_embedding_jit
        else:
            forward_pair_embedding_fn = _forward_pair_embedding_eager
        return forward_pair_embedding_fn(
            target_feat,
            residue_index,
            self.linear_tf_z_i.weight,
            self.linear_tf_z_i.bias,
            self.linear_tf_z_j.weight,
            self.linear_tf_z_j.bias,
            self.linear_relpos.weight,
            self.linear_relpos.bias,
            self.relpos_k,
        )

    def _forward_msa_embedding(
        self,
        msa_feat: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> torch.Tensor:
        if inductor.is_enabled():
            forward_msa_embedding_fn = _forward_msa_embedding_jit
        else:
            forward_msa_embedding_fn = _forward_msa_embedding_eager
        return forward_msa_embedding_fn(
            msa_feat,
            self.linear_msa_m.weight,
            self.linear_msa_m.bias,
            target_feat,
            self.linear_tf_m.weight,
            self.linear_tf_m.bias,
        )


def _forward_relpos(
    residue_index: torch.Tensor,
    w_relpos: torch.Tensor,
    b_relpos: torch.Tensor,
    relpos_k: int,
) -> torch.Tensor:
    """Relative position encoding.

    Supplementary '1.5 Input embeddings': Algorithm 4.

    """
    bins = torch.arange(
        start=-relpos_k,
        end=relpos_k + 1,
        step=1,
        dtype=residue_index.dtype,
        device=residue_index.device,
    )
    relative_distances = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
    return F.linear(_one_hot_relpos(relative_distances, bins), w_relpos, b_relpos)


def _one_hot_relpos(
    relative_distances: torch.Tensor,
    bins: torch.Tensor,
) -> torch.Tensor:
    """One-hot encoding with nearest bin.

    Supplementary '1.5 Input embeddings': Algorithm 5.

    """
    indices = (relative_distances.unsqueeze(-1) - bins).abs().argmin(dim=-1)
    return F.one_hot(indices, num_classes=len(bins)).to(dtype=relative_distances.dtype)


def _forward_pair_embedding_eager(
    target_feat: torch.Tensor,
    residue_index: torch.Tensor,
    w_tf_z_i: torch.Tensor,
    b_tf_z_i: torch.Tensor,
    w_tf_z_j: torch.Tensor,
    b_tf_z_j: torch.Tensor,
    w_relpos: torch.Tensor,
    b_relpos: torch.Tensor,
    relpos_k: int,
) -> torch.Tensor:
    tf_emb_i = F.linear(target_feat, w_tf_z_i, b_tf_z_i)  # a_i
    # tf_emb_i: [batch, N_res, c_z]
    tf_emb_j = F.linear(target_feat, w_tf_z_j, b_tf_z_j)  # b_j
    # tf_emb_j: [batch, N_res, c_z]
    residue_index = residue_index.to(dtype=target_feat.dtype)
    pair_emb = _forward_relpos(residue_index, w_relpos, b_relpos, relpos_k)
    pair_emb = pair_emb + tf_emb_i.unsqueeze(-2)
    pair_emb = pair_emb + tf_emb_j.unsqueeze(-3)
    # pair_emb: [batch, N_res, N_res, c_z]
    return pair_emb


_forward_pair_embedding_jit = torch.compile(_forward_pair_embedding_eager)


def _forward_msa_embedding_eager(
    msa_feat: torch.Tensor,
    msa_w: torch.Tensor,
    msa_b: torch.Tensor,
    target_feat: torch.Tensor,
    tf_w: torch.Tensor,
    tf_b: torch.Tensor,
) -> torch.Tensor:
    l1 = F.linear(msa_feat, msa_w, msa_b)
    l2 = F.linear(target_feat, tf_w, tf_b).unsqueeze(-3)
    msa_emb = l1 + l2
    # msa_emb: [batch, N_clust, N_res, c_m]
    return msa_emb


_forward_msa_embedding_jit = torch.compile(_forward_msa_embedding_eager)


class InputEmbedderMultimer(nn.Module):
    """Input Embedder module.

    Multimer '7.5 Chain Relative Positional Encoding'.

    """

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        max_relative_index: int,
        use_chain_relative: bool,
        max_relative_chain: int,
    ) -> None:
        super().__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)

        self.max_relative_index = max_relative_index
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if self.use_chain_relative:
            self.no_bins = 2 * max_relative_index + 2 + 1 + 2 * max_relative_chain + 2
        else:
            self.no_bins = 2 * max_relative_index + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def forward(
        self,
        target_feat: torch.Tensor,
        residue_index: torch.Tensor,
        sym_id: torch.Tensor,
        asym_id: torch.Tensor,
        entity_id: torch.Tensor,
        msa_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tf_emb_i = self.linear_tf_z_i(target_feat)  # a_i
        # tf_emb_i: [batch, N_res, c_z]

        tf_emb_j = self.linear_tf_z_j(target_feat)  # b_j
        # tf_emb_j: [batch, N_res, c_z]

        pair_emb = self.relpos(residue_index, sym_id, asym_id, entity_id)
        pair_emb = pair_emb + tf_emb_i.unsqueeze(-2)
        pair_emb = pair_emb + tf_emb_j.unsqueeze(-3)
        # pair_emb: [batch, N_res, N_res, c_z]

        msa_emb = self.linear_msa_m(msa_feat)
        msa_emb = msa_emb + self.linear_tf_m(target_feat).unsqueeze(-3)
        # msa_emb: [batch, N_clust, N_res, c_m]

        return msa_emb, pair_emb

    def relpos(
        self,
        residue_index: torch.Tensor,
        sym_id: torch.Tensor,
        asym_id: torch.Tensor,
        entity_id: torch.Tensor,
    ) -> torch.Tensor:
        asym_id_same = asym_id[..., :, None] == asym_id[..., None, :]
        offset = residue_index[..., :, None] - residue_index[..., None, :]

        clipped_offset = torch.clamp(offset + self.max_relative_index, 0, 2 * self.max_relative_index)

        rel_feats = []
        if self.use_chain_relative:
            final_offset = torch.where(
                asym_id_same,
                clipped_offset,
                (2 * self.max_relative_index + 1) * torch.ones_like(clipped_offset),
            )
            bins = torch.arange(0, 2 * self.max_relative_index + 2, device=final_offset.device)
            rel_pos = _one_hot_relpos(final_offset, bins)

            rel_feats.append(rel_pos)

            entity_id_same = entity_id[..., :, None] - sym_id[..., None, :]
            rel_feats.append(entity_id_same[..., None].to(dtype=rel_pos.dtype))

            rel_sym_id = sym_id[..., :, None] - sym_id[..., None, :]

            max_rel_chain = self.max_relative_chain
            clipped_rel_chain = torch.clamp(rel_sym_id + max_rel_chain, 0, 2 * max_rel_chain)

            final_rel_chain = torch.where(
                entity_id_same,
                clipped_rel_chain,
                (2 * max_rel_chain + 1) * torch.ones_like(clipped_rel_chain),
            )

            bins = torch.arange(0, 2 * max_rel_chain + 2, device=final_rel_chain.device)
            rel_chain = _one_hot_relpos(final_rel_chain, bins)

            rel_feats.append(rel_chain)
        else:
            bins = torch.arange(0, 2 * self.max_relative_index + 1, device=clipped_offset.device)
            rel_pos = _one_hot_relpos(clipped_offset, bins)
            rel_feats.append(rel_pos)

        rel_feat = torch.cat(rel_feats, dim=-1).to(self.linear_relpos.weight.dtype)

        return self.linear_relpos(rel_feat)
