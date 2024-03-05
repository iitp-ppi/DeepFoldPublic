import torch
import torch.nn as nn

import deepfold.model.v2.modules.geometry as geometry
from deepfold.model.v2.modules.layer_norm import LayerNorm
from deepfold.model.v2.modules.linear import Linear


class TemplatePairEmbedder(nn.Module):
    """Template Pair Embedder module.

    Embeds the "template_pair_feat" feature.

    Supplementary '1.4 AlphaFold Inference': Algorithm 2, line 9.

    Args:
        tp_dim: Input `template_pair_feat` dimension (channels).
        c_t: Output template representation dimension (channels).

    """

    def __init__(
        self,
        tp_dim: int,
        c_t: int,
    ) -> None:
        super().__init__()
        self.tp_dim = tp_dim
        self.c_t = c_t
        self.linear = Linear(tp_dim, c_t, bias=True, init="relu")

    def forward(
        self,
        template_pair_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Template Pair Embedder forward pass.

        Args:
            template_pair_feat: [batch, N_res, N_res, tp_dim]

        Returns:
            template_pair_embedding: [batch, N_res, N_res, c_t]

        """
        return self.linear(template_pair_feat)


class TemplatePairEmbedderMultimer(nn.Module):
    """Template Pair Embedder module for multimer model.

    Embeds the "template_pair_feat" feature.

    Args:
        tp_dim: Input `template_pair_feat` dimension (channels).
        c_t: Output template representation dimension (channels).

    """

    def __init__(
        self,
        c_z: int,
        c_t: int,
        c_dgram: int,
        c_aatype: int,
    ) -> None:
        super().__init__()
        self.c_z = c_z
        self.c_t = c_t
        self.c_dgram = c_dgram
        self.c_aatype = c_aatype

        self.dgram_linear = Linear(c_dgram, c_t, init="relu")
        self.aatype_linear_1 = Linear(c_aatype, c_t, init="relu")
        self.aatype_linear_2 = Linear(c_aatype, c_t, init="relu")

        self.query_embedding_layer_norm = LayerNorm(c_z)
        self.query_embedding_linear = Linear(c_z, c_t, init="relu")

        self.pseudo_beta_mask_linear = Linear(1, c_t, init="relu")
        self.x_linear = Linear(1, c_t, init="relu")
        self.y_linear = Linear(1, c_t, init="relu")
        self.z_linear = Linear(1, c_t, init="relu")
        self.backbone_mask_linear = Linear(1, c_t, init="relu")

    def forward(
        self,
        template_dgram: torch.Tensor,
        aatype_one_hot: torch.Tensor,
        query_embedding: torch.Tensor,  # z: pair representation
        pseudo_beta_mask: torch.Tensor,
        backbone_mask: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
        unit_vector: geometry.Vec3Array,
    ) -> torch.Tensor:
        act = 0.0

        pseudo_beta_mask_2d = pseudo_beta_mask[..., :, None] * pseudo_beta_mask[..., None, :]
        pseudo_beta_mask_2d *= multichain_mask_2d
        template_dgram *= pseudo_beta_mask_2d[..., None]
        act += self.dgram_linear(template_dgram)
        act += self.pseudo_beta_mask_linear(pseudo_beta_mask_2d[..., None])

        aatype_one_hot = aatype_one_hot.to(template_dgram.dtype)
        act += self.aatype_linear_1(aatype_one_hot[..., None, :, :])
        act += self.aatype_linear_2(aatype_one_hot[..., None, :])

        backbone_mask_2d = backbone_mask[..., :, None] * backbone_mask[..., None, :]
        backbone_mask_2d *= multichain_mask_2d
        x, y, z = [(coord * backbone_mask_2d).to(dtype=query_embedding.dtype) for coord in unit_vector]
        act += self.x_linear(x[..., None])
        act += self.y_linear(y[..., None])
        act += self.z_linear(z[..., None])

        act += self.backbone_mask_linear(backbone_mask_2d[..., None].to(dtype=query_embedding.dtype))

        query_embedding = self.query_embedding_layer_norm(query_embedding)
        act += self.query_embedding_linear(query_embedding)

        return act
