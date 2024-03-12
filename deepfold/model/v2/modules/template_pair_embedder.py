from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.common.residue_constants as rc
import deepfold.model.v2.modules.inductor as inductor
from deepfold.model.v2.modules.geometry import Rigid, Vector
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

    def build_template_pair_feat(
        self,
        feats: Dict[str, torch.Tensor],
        min_bin: int,
        max_bin: int,
        num_bins: int,
        use_unit_vector: bool,
        inf: float,
        eps: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        template_pseudo_beta = feats["template_pseudo_beta"]
        template_pseudo_beta_mask = feats["template_pseudo_beta_mask"]
        template_aatype = feats["template_aatype"]
        template_all_atom_mask = feats["template_all_atom_mask"]

        self._initialize_buffers(
            min_bin=min_bin,
            max_bin=max_bin,
            num_bins=num_bins,
            inf=inf,
            device=template_pseudo_beta.device,
        )

        if inductor.is_enabled():
            compute_part1_fn = _compute_part1_jit
        else:
            compute_part1_fn = _compute_part1_eager
        to_concat, aatype_one_hot = compute_part1_fn(
            template_pseudo_beta,
            template_pseudo_beta_mask,
            template_aatype,
            self.lower,
            self.upper,
            rc.RESTYPE_NUM + 2,
        )

        N_res = template_aatype.shape[-1]

        to_concat.append(aatype_one_hot.unsqueeze(-3).expand(*aatype_one_hot.shape[:-2], N_res, -1, -1))

        to_concat.append(aatype_one_hot.unsqueeze(-2).expand(*aatype_one_hot.shape[:-2], -1, N_res, -1))

        n, ca, c = [rc.ATOM_ORDER[a] for a in ["N", "CA", "C"]]

        if inductor.is_enabled():
            make_transform_from_reference = Rigid.make_transform_from_reference_jit
        else:
            make_transform_from_reference = Rigid.make_transform_from_reference
        rigids = make_transform_from_reference(
            n_xyz=feats["template_all_atom_positions"][..., n, :],
            ca_xyz=feats["template_all_atom_positions"][..., ca, :],
            c_xyz=feats["template_all_atom_positions"][..., c, :],
            eps=eps,
        )

        points = rigids.get_trans().unsqueeze(-3)
        rigid_vec = rigids.unsqueeze(-1).invert_apply_jit(points)

        if inductor.is_enabled():
            compute_part2_fn = _compute_part2_jit
        else:
            compute_part2_fn = _compute_part2_eager
        t = compute_part2_fn(
            rigid_vec,
            eps,
            template_all_atom_mask,
            n,
            ca,
            c,
            use_unit_vector,
            to_concat,
            dtype,
        )
        return t

    def _initialize_buffers(
        self,
        min_bin: int,
        max_bin: int,
        num_bins: int,
        inf: float,
        device: torch.device,
    ) -> None:
        if not hasattr(self, "lower") or not hasattr(self, "upper"):
            bins = torch.linspace(
                start=min_bin,
                end=max_bin,
                steps=num_bins,
                device=device,
                requires_grad=False,
            )
            lower = torch.pow(bins, 2)
            upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
            self.register_buffer("lower", lower, persistent=False)
            self.register_buffer("upper", upper, persistent=False)


def _compute_part1_eager(
    tpb: torch.Tensor,
    template_mask: torch.Tensor,
    template_aatype: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_classes: int,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    template_mask_2d = template_mask.unsqueeze(-1) * template_mask.unsqueeze(-2)
    dgram = torch.sum(
        input=(tpb.unsqueeze(-2) - tpb.unsqueeze(-3)) ** 2,
        dim=-1,
        keepdim=True,
    )
    dgram = ((dgram > lower) * (dgram < upper)).to(dtype=dgram.dtype)
    to_concat = [dgram, template_mask_2d.unsqueeze(-1)]
    aatype_one_hot = F.one_hot(
        input=template_aatype,
        num_classes=num_classes,
    )
    return to_concat, aatype_one_hot


_compute_part1_jit = torch.compile(_compute_part1_eager)


def _compute_part2_eager(
    rigid_vec: torch.Tensor,
    eps: float,
    t_aa_masks: torch.Tensor,
    n: int,
    ca: int,
    c: int,
    use_unit_vector: bool,
    to_concat: List[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec**2, dim=-1))
    template_mask = t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[..., c]
    template_mask_2d = template_mask.unsqueeze(-1) * template_mask.unsqueeze(-2)
    inv_distance_scalar = inv_distance_scalar * template_mask_2d
    unit_vector = rigid_vec * inv_distance_scalar.unsqueeze(-1)
    if not use_unit_vector:
        unit_vector = unit_vector * 0.0
    to_concat.extend(torch.unbind(unit_vector.unsqueeze(-2), dim=-1))
    to_concat.append(template_mask_2d.unsqueeze(-1))
    t = torch.cat(to_concat, dim=-1)
    t = t * template_mask_2d.unsqueeze(-1)
    t = t.to(dtype=dtype)
    return t


_compute_part2_jit = torch.compile(_compute_part2_eager)


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
        unit_vector: Vector,
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
