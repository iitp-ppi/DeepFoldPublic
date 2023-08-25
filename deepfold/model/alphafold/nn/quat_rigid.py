# DeepFold Team

import torch
import torch.nn as nn

from deepfold.model.alphafold.nn.primitives import Linear
from deepfold.utils.geometry import Rigid3Array, Rot3Array, Vec3Array


class QuatRigid(nn.Module):
    def __init__(self, c_hidden: int, full_quat: bool):
        super().__init__()
        self.full_quat = full_quat
        if self.full_quat:
            rigid_dim = 7
        else:
            rigid_dim = 6

        self.linear = Linear(c_hidden, rigid_dim, init="final")

    def forward(self, act: torch.Tensor) -> Rigid3Array:
        rigid_flat = self.linear(act)
        rigid_flat = torch.unbind(rigid_flat, dim=-1)
        if self.full_quat:
            qw, qx, qy, qz = rigid_flat[:4]
            translation = rigid_flat[4:]
        else:
            qx, qy, qz = rigid_flat[:3]
            qw = torch.ones_like(qx)
            translation = rigid_flat[3:]

        rotation = Rot3Array.from_quaternion(qw, qx, qy, qz, normalize=True)
        translation = Vec3Array(*translation)

        return Rigid3Array(rotation, translation)
