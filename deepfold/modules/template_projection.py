import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.modules.inductor as inductor
from deepfold.modules.linear import Linear


class TemplateProjection(nn.Module):
    """Template Projection module.

    Multimer '7.7. Architectural Modifications'.

    Args:
        c_t: Template representation dimension (channels).
        c_z: Pair representation dimension (channels).

    """

    def __init__(self, c_t: int, c_z: int) -> None:
        super().__init__()
        self.linear_t = Linear(c_t, c_z, bias=True, init="default")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Template Projector forward pass.

        Args:
            t: [batch, N_templ, N_res, N_res, c_t] template representation

        Returns:
            z_update: [batch, N_res, N_res, c_z] pair representation update
                from template representation

        """
        # Average template features.
        t = torch.mean(t, dim=-4)
        # t: [*, N_res, N_res, c_t]

        if inductor.is_enabled():
            z_update = _forward_template_projection_jit(t, self.linear_t.weight, self.linear_t.bias)
        else:
            z_update = _forward_template_projection_eager(t, self.linear_t.weight, self.linear_t.bias)
        # z_update: [batch, N_res, N_res, c_z]

        return z_update


def _forward_template_projection_eager(
    t: torch.Tensor,
    w_proj: torch.Tensor,
    b_proj: torch.Tensor,
) -> torch.Tensor:
    return F.linear(F.relu(t), w_proj, b_proj)


_forward_template_projection_jit = torch.compile(_forward_template_projection_eager)
