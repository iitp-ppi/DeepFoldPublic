# DeepFold Team


from __future__ import annotations

import dataclasses
from typing import List

import numpy as np
import torch

from deepfold.utils.geometry.vector import Vec3Array
from deepfold.utils.misc import get_field_names
from deepfold.utils.tensor_utils import tensor_tree_map

COMPONENTS: List[str] = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]


@dataclasses.dataclass(frozen=True)
class Rot3Array:
    xx: torch.Tensor = dataclasses.field(metadata={"dtype": torch.float32})
    xy: torch.Tensor
    xz: torch.Tensor
    yx: torch.Tensor
    yy: torch.Tensor
    yz: torch.Tensor
    zx: torch.Tensor
    zy: torch.Tensor
    zz: torch.Tensor

    __array_ufunc__ = None

    def __getitem__(self, index) -> Rot3Array:
        field_names = get_field_names(Rot3Array)

        return Rot3Array(**{name: getattr(self, name)[index] for name in field_names})

    def __mul__(self, other: torch.Tensor) -> Rot3Array:
        field_names = get_field_names(Rot3Array)

        return Rot3Array(**{name: getattr(self, name) * other for name in field_names})

    def __matmul__(self, other: Rot3Array) -> Rot3Array:
        c0 = self.apply_to_point(Vec3Array(other.xx, other.yx, other.zx))
        c1 = self.apply_to_point(Vec3Array(other.xy, other.yy, other.zy))
        c2 = self.apply_to_point(Vec3Array(other.xz, other.yz, other.zz))

        return Rot3Array(c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z)

    def map_tensor_fn(self, fn) -> Rot3Array:
        field_names = get_field_names(Rot3Array)

        return Rot3Array(**{name: fn(getattr(self, name)) for name in field_names})

    def inverse(self) -> Rot3Array:
        return Rot3Array(self.xx, self.yx, self.zx, self.xy, self.yy, self.zy, self.xz, self.yz, self.zz)

    def apply_to_point(self, point: Vec3Array) -> Vec3Array:
        return Vec3Array(
            self.xx * point.x + self.xy * point.y + self.xz * point.z,
            self.yx * point.x + self.yy * point.y + self.yz * point.z,
            self.zx * point.x + self.zy * point.y + self.zz * point.z,
        )

    def apply_inverse_to_point(self, point: Vec3Array) -> Vec3Array:
        return self.inverse().apply_to_point(point)

    def unsqueeze(self, dim: int) -> Rot3Array:
        return Rot3Array(*tensor_tree_map(lambda t: t.unsqueeze(dim), [getattr(self, c) for c in COMPONENTS]))

    def stop_gradient(self) -> Rot3Array:
        return Rot3Array(*[getattr(self, c).detach() for c in COMPONENTS])

    @classmethod
    def identity(cls, shape, device) -> Rot3Array:
        ones = torch.ones(shape, dtype=torch.float32, device=device)
        zeros = torch.zeros(shape, dtype=torch.float32, device=device)

        return cls(ones, zeros, zeros, zeros, ones, zeros, zeros, zeros, ones)

    @classmethod
    def from_two_vectors(cls, e0: Vec3Array, e1: Vec3Array) -> Rot3Array:
        """
        Construct Rot3Array from two vectors.

        Rot3Array is constructed such that in the corresponding frame `e0` lies on the postiive x-axis
        and `e1` lies in the xy plane with positive sign of y.
        """

        # Normalize
        e0 = e0.normalized()
        # Project
        c = e1.dot(e0)
        e1 = (e1 - c * e0).normalized()

        e2 = e0.cross(e1)

        return cls(e0.x, e1.x, e2.x, e0.y, e1.y, e2.y, e0.z, e1.z, e2.z)

    @classmethod
    def from_array(cls, tensor: torch.Tensor) -> Rot3Array:
        """Construct Rot3Array from array of shape [..., 3, 3]."""
        rows = torch.unbind(tensor, dim=-2)
        rc = [torch.unbind(e, dim=-1) for e in rows]

        return cls(*[e for row in rc for e in row])

    @classmethod
    def from_quaternion(
        cls,
        w: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        normalize: bool = True,
        eps: float = 1e-6,
    ) -> Rot3Array:
        if normalize:
            inv_norm = torch.rsqrt(torch.clamp(w**2 + x**2 + y**2 + z**2, min=eps))
            w = w * inv_norm
            x = x * inv_norm
            y = y * inv_norm
            z = z * inv_norm
        xx = 1.0 - 2.0 * (y**2 + z**2)
        xy = 2.0 * (x * y - w * z)
        xz = 2.0 * (x * z + w * y)
        yx = 2.0 * (x * y + w * z)
        yy = 1.0 - 2.0 * (x**2 + z**2)
        yz = 2.0 * (y * z - w * x)
        zx = 2.0 * (x * z - w * y)
        zy = 2.0 * (y * z + w * x)
        zz = 1.0 - 2.0 * (x**2 + y**2)
        return cls(xx, xy, xz, yx, yy, yz, zx, zy, zz)

    def reshape(self, new_shape) -> Rot3Array:
        field_names = get_field_names(Rot3Array)
        reshape_fn = lambda t: t.reshape(new_shape)

        return Rot3Array(**{name: reshape_fn(getattr(self, name)) for name in field_names})

    @classmethod
    def cat(cls, rots: List[Rot3Array], dim: int) -> Rot3Array:
        field_names = get_field_names(Rot3Array)
        cat_fn = lambda l: torch.cat(l, dim=dim)
        return cls(**{name: cat_fn([getattr(r, name) for r in rots]) for name in field_names})
