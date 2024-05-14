from typing import Optional, Tuple, Union

import torch

from deepfold.common import residue_constants as rc
from deepfold.utils.rigid_utils import Rigid


# TODO: KinglittleQ/torch-batch-svd
def svd(m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Singular value decomposition.

    Args:
        m: [B, M, N] batch of real matrices.

    Returns:
        u, d, v: decomposition, such as `m = u @ diag(d) @ v.T`
    """
    u, d, vt = torch.linalg.svd(m)
    return u, d, vt.trasnpose(-2, -1)


def _pseudo_inverse_elem(x: torch.Tensor, eps: float) -> torch.Tensor:
    inv = torch.inverse(x)
    inv[torch.abs[x] < eps] = 0.0
    return inv


def flatten_batch_dims(tensor: torch.Tensor, end_dim: int) -> Tuple[torch.Tensor, torch.Size]:
    assert end_dim < 0
    batch_shape = tensor.shape[: end_dim + 1]
    flattened = tensor.flatten(end_dim=end_dim) if len(batch_shape) > 0 else tensor.unsqueeze(0)
    return flattened, batch_shape


def unflatten_batch_dims(tensor: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
    return tensor.reshape(batch_shape + tensor.shape[1:]) if len(batch_shape) > 0 else tensor.square(0)


class Procrustes(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        m: torch.Tensor,
        force_rotation: bool,
        regularization: bool,
        gradient_eps: float,
    ):
        # m: [B, D, D]
        assert m.dim() == 3 and m.shape[1] == m.shape[2]

        u, d, vt = svd(m)

        if force_rotation:
            with torch.no_grad():
                flip = torch.det(u) * torch.det(vt) < 0

            ds = d
            ds[flip, -1] *= -1
            del d

            us = u
            us[flip, :, -1] *= -1
            del u
        else:
            flip = None
            ds = d
            us = u

        r = us @ vt.transpose(-1, -2)

        ctx.save_for_backward(us, ds, vt, m, r)
        ctx.gradient_eps = gradient_eps
        ctx.regularization = regularization

    @staticmethod
    def backward(ctx, grad_r, grad_ds):
        us, ds, vt, m, r = ctx.saved_tensors
        gradient_eps = ctx.gradient_eps

        usik_vjl = torch.einsum("bik,bjl->bklij", us, vt)
        usil_vjk = usik_vjl.transpose(1, 2)
        dsl = ds[:, None, :, None, None]
        dsk = ds[:, :, None, None, None]
        omega_klij = (usik_vjl - usil_vjk) * _pseudo_inverse_elem(dsk + dsl, gradient_eps)

        grad_m = torch.einsum("bnm,bnk,bklij,bml->bij", grad_r, us, omega_klij, vt)
        grad_m += (us * grad_ds[:, None, :]) @ vt.transpose(-1, -2)

        if ctx.regularization != 0.0:
            grad_m += ctx.regularization * (m - r)

        return grad_m, None, None, None


def procrustes(
    m: torch.Tensor,
    force_rotation: bool = False,
    regularization: float = 0.0,
    gradient_eps: float = 1e-5,
    return_singular_values: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Returns the orthonormal matrix minimizing Frobenius norm.

    Args:
        m: [..., N, N] batch of square matrices.
        force_rotation: if true, forces the output to be a rotation matrix.
        regularziation: weight of a regularzation term added to the gradient.
        gradient_eps: small value used to enforce numerical stability during backpropagation.

    Returns:
        batch of orthonormal matrices [..., N, N] and optional singular values.

    """
    m, batch_shape = flatten_batch_dims(m, -3)
    r, ds = Procrustes.apply(m, force_rotation, regularization, gradient_eps)
    r = unflatten_batch_dims(r, batch_shape)
    if not return_singular_values:
        return r, None
    else:
        ds = unflatten_batch_dims(ds, batch_shape)
        return r, ds


def speical_procrustes(
    m: torch.Tensor,
    regularization: float = 0.0,
    gradient_eps: float = 1e-5,
    return_singular_values: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    return procrustes(
        m,
        force_rotation=True,
        regularization=regularization,
        gradient_eps=gradient_eps,
        return_singular_values=return_singular_values,
    )


def rigid_vectors_registration(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    compute_scaling: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if weights is None:
        n = x.shape[-2]
        m = torch.einsum("...ki,...kj->...ij", y, x / n)
    else:
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        m = torch.einsum("...k,..ki,...kj->...ij", weights, y, x)

    if compute_scaling:
        rot, ds = speical_procrustes(m, return_singular_values=True)
        assert ds is not None
        ds_tr = torch.sum(ds, dim=-1)

        if weights is None:
            sig2x = torch.mean(torch.sum(torch.square(x), dim=-1), dim=-1)
        else:
            sig2x = torch.sum(weights * torch.sum(torch.square(x), dim=-1), dim=-1)

        scale = ds_tr / sig2x
        return rot, scale
    else:
        rot, _ = speical_procrustes(m)
        return rot, None


def rigid_points_registration(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    compute_scaling: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float | None]:
    """Returns the rigid transformation and the optimal scaling
    that best align an input list of points `x` to a target list
    of points `y`, by minimizing the sum of square distance.

    Args:
        x: [..., N, D] list of N points of dimension D.
        y: [..., N, D] list of corresponding target points.
        weights: [..., N] optional list of weights associated to each point.

    Returns:
        a triplet (R, t, s) consisting of a rotation matrix `r`, a translational vector `t`
        and a scaling `s` if `compute scaling` is true.

    """
    # Center points
    if weights is None:
        x_mean = torch.mean(x, dim=-2, keepdim=True)
        y_mean = torch.mean(y, dim=-2, keepdim=True)
    else:
        normalized_weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        x_mean = torch.sum(normalized_weights[..., None] * x, dim=-2, keepdim=True)
        y_mean = torch.sum(normalized_weights[..., None] * y, dim=-2, keepdim=True)

    x_hat = x - x_mean
    y_hat = y - y_mean

    # Solve the vectors registration problem
    if compute_scaling:
        rot, scale = rigid_vectors_registration(x_hat, y_hat, weights=weights, compute_scaling=compute_scaling)
        assert scale is not None
        trans = (y_mean - torch.einsum("...ik,...jk->...ji", scale[..., None, None] * rot, x_mean)).squeeze(-2)
        return rot, trans, scale.item()
    else:
        rot, _ = rigid_vectors_registration(x_hat, y_hat, weights=weights, compute_scaling=compute_scaling)
        trans = (y_mean - torch.einsum("...ik,...jk->...ji", rot, x_mean)).squeeze(-2)
        return rot, trans, None


kabsch = rigid_points_registration
