import torch
import torch.nn as nn
import torch.nn.functional as F

import deepfold.modules.inductor as inductor
from deepfold.modules.linear import Linear


class ExtraMSAEmbedder(nn.Module):
    """Extra MSA Embedder module.

    Embeds the "extra_msa_feat" feature.

    Supplementary '1.4 AlphaFold Inference': Algorithm 2, line 15.

    Args:
        emsa_dim: Input `extra_msa_feat` dimension (channels).
        c_e: Output extra MSA representation dimension (channels).

    """

    def __init__(
        self,
        emsa_dim: int,
        c_e: int,
    ) -> None:
        super().__init__()
        self.linear = Linear(emsa_dim, c_e, bias=True, init="default")

    def forward(
        self,
        extra_msa_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Extra MSA Embedder forward pass.

        Args:
            extra_msa_feat: [batch, N_extra_seq, N_res, emsa_dim]

        Returns:
            extra_msa_embedding: [batch, N_extra_seq, N_res, c_e]

        """
        if inductor.is_enabled():
            forward_fn = _forward_jit
        else:
            forward_fn = _forward_eager
        return forward_fn(
            extra_msa_feat,
            self.linear.weight,
            self.linear.bias,
        )


def _forward_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, w, b)


_forward_jit = torch.compile(_forward_eager)
