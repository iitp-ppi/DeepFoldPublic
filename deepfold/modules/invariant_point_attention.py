import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.modules.inductor as inductor
from deepfold.modules.linear import Linear
from deepfold.utils.geometry import Rigid3Array, Vec3Array, square_euclidean_distance
from deepfold.utils.precision import is_fp16_enabled
from deepfold.utils.rigid_utils import Rigid
from deepfold.utils.tensor_utils import flatten_final_dims


class InvariantPointAttention(nn.Module):
    """Invariant Point Attention (IPA) module.

    Supplementary '1.8.2 Invariant point attention (IPA)': Algorithm 22.

    Args:
        c_s: Single representation dimension (channels).
        c_z: Pair representation dimension (channels).
        c_hidden: Hidden dimension (channels).
        num_heads: Number of attention heads.
        num_qk_points: Number of query/key points.
        num_v_points: Number of value points.
        is_multimer: Separate key/value projection.
        inf: Safe infinity value.
        eps: Epsilon to prevent division by zero.

    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        num_heads: int,
        num_qk_points: int,
        num_v_points: int,
        separate_kv: bool,
        inf: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.separate_kv = separate_kv
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the supplement.
        # There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias
        # and use the default Lecun initialization.
        hc = c_hidden * num_heads
        self.linear_q = Linear(c_s, hc, bias=True, init="default")
        if self.separate_kv:
            self.linear_k = Linear(c_s, hc, bias=True, init="default")
            self.linear_v = Linear(c_s, hc, bias=True, init="default")
        else:
            self.linear_kv = Linear(c_s, 2 * hc, bias=True, init="default")

        hpq = num_heads * num_qk_points * 3
        self.linear_q_points = Linear(c_s, hpq, bias=True, init="default")
        hpk = self.num_heads * self.num_qk_points * 3
        hpv = self.num_heads * self.num_v_points * 3
        if self.separate_kv:
            self.linear_k_points = Linear(c_s, hpk, bias=True, init="default")
            self.linear_v_points = Linear(c_s, hpv, bias=True, init="default")
        else:
            hpkv = hpk + hpv
            self.linear_kv_points = Linear(c_s, hpkv, bias=True, init="default")

        self.linear_b = Linear(c_z, num_heads, bias=True, init="default")

        self.head_weights = nn.Parameter(torch.zeros((num_heads)))
        ipa_point_weights_init_(self.head_weights.data)

        concat_out_dim = num_heads * (c_z + c_hidden + num_v_points * 4)
        self.linear_out = Linear(concat_out_dim, c_s, bias=True, init="final")

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Invariant Point Attention (IPA) forward pass.

        Args:
            s: [batch, N_res, c_s] single representation
            z: [batch, N_res, N_res, c_z] pair representation
            r: [batch, N_res] rigids transformation
            mask: [batch, N_res] sequence mask

        Returns:
            s_update: [batch, N_res, c_s] single representation update

        """
        #######################################
        # Generate scalar and point activations
        #######################################
        if self.separate_kv:  # Multimer
            q = self.linear_q(s)
            bias = self.linear_b(z)
            k = self.linear_k(s)
            v = self.linear_v(s)
            q_pts = self.linear_q_points(s)
            k_pts = self.linear_k_points(s)
            v_pts = self.linear_v_points(s)
        else:
            q, bias, kv, q_pts, kv_pts = _forward_linears_on_inputs_eager(
                s,
                z,
                self.linear_q.weight,
                self.linear_q.bias,
                self.linear_b.weight,
                self.linear_b.bias,
                self.linear_kv.weight,
                self.linear_kv.bias,
                self.linear_q_points.weight,
                self.linear_q_points.bias,
                self.linear_kv_points.weight,
                self.linear_kv_points.bias,
            )
        # q: [batch, N_res, num_heads * c_hidden]
        # b: [batch, N_res, N_res, num_heads]
        # kv: [batch, N_res, num_heads * 2 * c_hidden]
        # k/v: [batch, N_res, num_heads * c_hidden]
        # q_pts: [batch, N_res, num_heads * num_qk_points * 3]
        # kv_pts: [batch, N_res, num_heads * (num_qk_points + num_v_points) * 3]

        q = q.view(q.shape[:-1] + (self.num_heads, self.c_hidden))
        # q: [batch, N_res, num_heads, c_hidden]

        if self.separate_kv:
            k = k.view(k.shape[:-1] + (self.num_heads, -1))
            v = v.view(v.shape[:-1] + (self.num_heads, -1))
        else:
            kv = kv.view(kv.shape[:-1] + (self.num_heads, -1))
            # kv: [batch, N_res, num_heads, 2 * c_hidden]
            k, v = torch.split(kv, self.c_hidden, dim=-1)
        # k: [batch, N_res, num_heads, c_hidden]
        # v: [batch, N_res, num_heads, c_hidden]

        def process_points(pts: torch.Tensor, num_points: int) -> torch.Tensor:
            shape = pts.shape[:-1] + (pts.shape[-1] // 3, 3)
            if self.separate_kv:
                pts = pts.view(pts.shape[:-1] + (self.num_heads, num_points * 3))
            pts = torch.split(pts, pts.shape[-1] // 3, dim=-1)
            pts = torch.stack(pts, dim=-1).view(*shape)
            pts = r[..., None].apply(pts)
            return pts.view(pts.shape[:-2] + (self.num_heads, num_points, 3))

        q_pts = process_points(q_pts, self.num_qk_points)
        # q_pts: [batch, N_res, num_heads, num_qk_points, 3]

        if self.separate_kv:
            k_pts = process_points(k_pts, self.num_qk_points)
            v_pts = process_points(v_pts, self.num_v_points)
            # k_pts: [batch, N_res, num_heads, num_qk_points, 3]
            # v_pts: [batch, N_res, num_heads, num_v_points, 3]
        else:
            kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
            kv_pts = torch.stack(kv_pts, dim=-1)
            kv_pts = r[..., None].apply(kv_pts)
            # kv_pts: [batch, N_res, num_heads * (num_qk_points + num_v_points), 3]

            kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.num_heads, self.num_qk_points + self.num_v_points, 3))
            # kv_pts: [batch, N_res, num_heads, (num_qk_points + num_v_points), 3]

            k_pts, v_pts = torch.split(kv_pts, (self.num_qk_points, self.num_v_points), dim=-2)
            # k_pts: [batch, N_res, num_heads, num_qk_points, 3]
            # v_pts: [batch, N_res, num_heads, num_v_points, 3]

        ##########################
        # Compute attention scores
        ##########################

        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                a = torch.matmul(
                    q.float().movedim(-3, -2),  # q: [batch, num_heads, N_res, c_hidden]
                    k.float().movedim(-3, -1),  # k: [batch, num_heads, c_hidden, N_res]
                )
        else:
            a = torch.matmul(
                q.movedim(-3, -2),  # q: [batch, num_heads, N_res, c_hidden]
                k.movedim(-3, -1),  # k: [batch, num_heads, c_hidden, N_res]
            )
        # a: [batch, num_heads, N_res, N_res]

        if inductor.is_enabled():
            forward_a_fn = _forward_a_jit
        else:
            forward_a_fn = _forward_a_eager
        a = forward_a_fn(
            a,
            bias,
            q_pts,
            k_pts,
            mask,
            self.c_hidden,
            self.num_heads,
            self.head_weights,
            self.num_qk_points,
            self.inf,
        )
        # a: [batch, num_heads, N_res, N_res]

        ################
        # Compute output
        ################

        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)
        # o: [batch, N_res, num_heads, c_hidden]

        o = o.reshape(o.shape[:-2] + (self.num_heads * self.c_hidden,))
        # o: [batch, N_res, num_heads * c_hidden]

        o_pt = torch.sum((a.unsqueeze(-3).unsqueeze(-1) * v_pts.swapdims(-4, -3).movedim(-1, -3).unsqueeze(-3)), dim=-2)
        # o_pt: [batch, num_heads, 3, N_res, num_v_points]

        o_pt = o_pt.movedim(-3, -1).swapdims(-3, -4)
        # o_pt: [batch, N_res, num_heads, num_v_points, 3]

        o_pt = r.unsqueeze(-1).unsqueeze(-2).invert_apply(o_pt)
        # o_pt: [batch, N_res, num_heads, num_v_points, 3]

        if inductor.is_enabled():
            forward_o_pt_norm_fn = _forward_o_pt_norm_jit
        else:
            forward_o_pt_norm_fn = _forward_o_pt_norm_eager
        o_pt_norm = forward_o_pt_norm_fn(o_pt, self.eps)
        # o_pt_norm: [batch, N_res, num_heads, num_v_points]

        o_pt_norm = o_pt_norm.reshape(o_pt_norm.shape[:-2] + (self.num_heads * self.num_v_points,))
        # o_pt_norm: [batch, N_res, num_heads * num_v_points]

        o_pt = o_pt.reshape(o_pt.shape[:-3] + (self.num_heads * self.num_v_points, 3))
        # o_pt: [batch, N_res, num_heads * num_v_points, 3]

        o_pair = torch.matmul(a.transpose(-2, -3), z.to(dtype=a.dtype))
        # o_pair: [batch, N_res, num_heads, c_z]

        o_pair = o_pair.reshape(o_pair.shape[:-2] + (self.num_heads * self.c_z,))
        # o_pair: [batch, N_res, num_heads * c_z]

        o_cat = (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair)
        o_cat = torch.cat(o_cat, dim=-1)
        # o_cat: [batch, N_res, num_heads * (c_hidden + num_v_points * 4 + c_z)]

        s_update = self.linear_out(o_cat.to(dtype=z.dtype))
        # s_update: [batch, N_res, c_s]

        return s_update


def ipa_point_weights_init_(weights_data: torch.Tensor) -> None:
    softplus_inverse_1 = 0.541324854612918
    weights_data.fill_(softplus_inverse_1)


def _forward_linears_on_inputs_eager(
    s: torch.Tensor,
    z: torch.Tensor,
    w_q: torch.Tensor,
    b_q: torch.Tensor,
    w_b: torch.Tensor,
    b_b: torch.Tensor,
    w_kv: torch.Tensor,
    b_kv: torch.Tensor,
    w_q_points: torch.Tensor,
    b_q_points: torch.Tensor,
    w_kv_points: torch.Tensor,
    b_kv_points: torch.Tensor,
) -> torch.Tensor:
    q = F.linear(s, w_q, b_q)
    b = F.linear(z, w_b, b_b)
    kv = F.linear(s, w_kv, b_kv)
    q_pts = F.linear(s, w_q_points, b_q_points)
    kv_pts = F.linear(s, w_kv_points, b_kv_points)
    return q, b, kv, q_pts, kv_pts


_forward_linears_on_inputs_jit = torch.compile(_forward_linears_on_inputs_eager)


def _forward_a_eager(
    a: torch.Tensor,
    b: torch.Tensor,
    q_pts: torch.Tensor,
    k_pts: torch.Tensor,
    mask: torch.Tensor,
    c_hidden: int,
    num_heads: int,
    head_weights: torch.Tensor,
    num_qk_points: int,
    inf: float,
) -> torch.Tensor:
    # a: [batch, num_heads, N_res, N_res]
    # b: [batch, N_res, N_res, num_heads]
    # q_pts: [batch, N_res, num_heads, num_qk_points, 3]
    # k_pts: [batch, N_res, num_heads, num_qk_points, 3]
    # mask: [batch, N_res]

    a = a * math.sqrt(1.0 / (3 * c_hidden))
    a = a + (math.sqrt(1.0 / 3) * b.movedim(-1, -3))
    # a: [batch, num_heads, N_res, N_res]

    pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)  # outer subtraction
    pt_att = pt_att**2
    # pt_att: [batch, N_res, N_res, num_heads, num_qk_points, 3]

    pt_att = sum(torch.unbind(pt_att, dim=-1))
    # pt_att: [batch, N_res, N_res, num_heads, num_qk_points]

    head_weights = F.softplus(head_weights)
    head_weights = head_weights.view((1,) * (pt_att.ndim - 2) + (num_heads, 1))
    head_weights = head_weights * math.sqrt(1.0 / (3 * (num_qk_points * 9.0 / 2)))
    # head_weights: [1, 1, 1, num_heads, 1]

    pt_att = pt_att * head_weights
    # pt_att: [batch, N_res, N_res, num_heads, num_qk_points]

    pt_att = -0.5 * torch.sum(pt_att, dim=-1)
    # pt_att: [batch, N_res, N_res, num_heads]

    square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # outer product
    square_mask = (square_mask - 1.0) * inf
    # square_mask: [batch, N_res, N_res]

    pt_att = pt_att.movedim(-1, -3)
    # square_mask: [batch, num_heads, N_res, N_res]

    a = a + pt_att
    # a: [batch, num_heads, N_res, N_res]

    a = a + square_mask.unsqueeze(-3)
    # a: [batch, num_heads, N_res, N_res]

    a = torch.softmax(a, dim=-1)
    # a: [batch, num_heads, N_res, N_res]

    return a


_forward_a_jit = torch.compile(_forward_a_eager)


def _forward_o_pt_norm_eager(o_pt: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(torch.sum(o_pt**2, dim=-1) + eps)


_forward_o_pt_norm_jit = torch.compile(_forward_o_pt_norm_eager)


class PointProjection(nn.Module):
    def __init__(
        self,
        c_hidden: int,
        num_points: int,
        no_heads: int,
        is_multimer: bool,
        return_local_points: bool = False,
    ):
        super().__init__()
        self.return_local_points = return_local_points
        self.no_heads = no_heads
        self.num_points = num_points
        self.is_multimer = is_multimer

        # TODO: Multimer requires this to be run with fp32 precision during training
        self.linear = Linear(c_hidden, no_heads * 3 * num_points)

    def forward(
        self,
        activations: torch.Tensor,
        rigids: Union[Rigid, Rigid3Array],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # TODO: Needs to run in high precision during training
        points_local = self.linear(activations)
        out_shape = points_local.shape[:-1] + (self.no_heads, self.num_points, 3)

        if self.is_multimer:
            points_local = points_local.view(points_local.shape[:-1] + (self.no_heads, -1))

        points_local = torch.split(points_local, points_local.shape[-1] // 3, dim=-1)

        points_local = torch.stack(points_local, dim=-1).view(out_shape)

        points_global = rigids[..., None, None].apply(points_local)

        if self.return_local_points:
            return points_global, points_local

        return points_global


class InvariantPointAttentionMultimer(nn.Module):

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        num_heads: int,
        num_qk_points: int,
        num_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            num_heads:
                Number of attention heads
            num_qk_points:
                Number of query/key points to generate
            num_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttentionMultimer, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the supplement.
        # There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default Lecun initialization.
        hc = self.c_hidden * self.num_heads
        self.linear_q = Linear(self.c_s, hc, bias=False)

        self.linear_q_points = PointProjection(self.c_s, self.num_qk_points, self.num_heads, is_multimer=True)

        self.linear_k = Linear(self.c_s, hc, bias=False)
        self.linear_v = Linear(self.c_s, hc, bias=False)
        self.linear_k_points = PointProjection(self.c_s, self.num_qk_points, self.num_heads, is_multimer=True)

        self.linear_v_points = PointProjection(self.c_s, self.num_v_points, self.num_heads, is_multimer=True)

        self.linear_b = Linear(self.c_z, self.num_heads)

        self.head_weights = nn.Parameter(torch.zeros((num_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.num_heads * (self.c_z + self.c_hidden + self.num_v_points * 4)
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-2)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid3Array,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """

        a = 0.0

        point_variance = max(self.num_qk_points, 1) * 9.0 / 2
        point_weights = math.sqrt(1.0 / point_variance)

        softplus = lambda x: torch.logaddexp(x, torch.zeros_like(x))

        head_weights = softplus(self.head_weights)
        point_weights = point_weights * head_weights

        #######################################
        # Generate scalar and point activations
        #######################################

        # [*, N_res, H, P_qk]
        q_pts = Vec3Array.from_array(self.linear_q_points(s, r))

        # [*, N_res, H, P_qk, 3]
        k_pts = Vec3Array.from_array(self.linear_k_points(s, r))

        pt_att = square_euclidean_distance(q_pts.unsqueeze(-3), k_pts.unsqueeze(-4), epsilon=0.0)
        pt_att = torch.sum(pt_att * point_weights[..., None], dim=-1) * (-0.5)
        pt_att = pt_att.to(dtype=s.dtype)
        a = a + pt_att

        scalar_variance = max(self.c_hidden, 1) * 1.0
        scalar_weights = math.sqrt(1.0 / scalar_variance)

        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        k = self.linear_k(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))
        k = k.view(k.shape[:-1] + (self.num_heads, -1))

        q = q * scalar_weights
        a = a + torch.einsum("...qhc,...khc->...qkh", q, k)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z)

        a = a + b

        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        a = a + square_mask.unsqueeze(-1)
        a = a * math.sqrt(1.0 / 3)  # Normalize by number of logit terms (3)
        a = self.softmax(a)

        # [*, N_res, H * C_hidden]
        v = self.linear_v(s)

        # [*, N_res, H, C_hidden]
        v = v.view(v.shape[:-1] + (self.num_heads, -1))

        o = torch.einsum("...qkh,...khc->...qhc", a, v)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, N_res, H, P_v, 3]
        v_pts = Vec3Array.from_array(self.linear_v_points(s, r))

        # [*, N_res, H, P_v]
        o_pt = v_pts[..., None, :, :, :] * a.unsqueeze(-1)
        o_pt = o_pt.sum(dim=-3)
        # o_pt = Vec3Array(
        #     torch.sum(a.unsqueeze(-1) * v_pts[..., None, :, :, :].x, dim=-3),
        #     torch.sum(a.unsqueeze(-1) * v_pts[..., None, :, :, :].y, dim=-3),
        #     torch.sum(a.unsqueeze(-1) * v_pts[..., None, :, :, :].z, dim=-3),
        # )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(o_pt.shape[:-2] + (-1,))

        # [*, N_res, H, P_v]
        o_pt = r[..., None].apply_inverse_to_point(o_pt)
        o_pt_flat = [o_pt.x, o_pt.y, o_pt.z]
        o_pt_flat = [x.to(dtype=a.dtype) for x in o_pt_flat]

        # [*, N_res, H * P_v]
        o_pt_norm = o_pt.norm(epsilon=1e-8)

        o_pair = torch.einsum("...ijh,...ijc->...ihc", a, z.to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(torch.cat((o, *o_pt_flat, o_pt_norm, o_pair), dim=-1).to(dtype=z.dtype))

        return s
