from typing import Optional

import torch
import torch.nn as nn

from deepfold.model.v2.modules.layer_norm import LayerNorm
from deepfold.model.v2.modules.linear import Linear
from deepfold.utils.chunk_utils import chunk_layer


class PairTransition(nn.Module):
    """Pair Transition module.

    Supplementary '1.6.7 Transition in the pair stack': Algorithm 15.

    Args:
        c_z: Pair or template representation dimension (channels).
        n: `c_z` multiplier to obtain hidden dimension (channels).

    """

    def __init__(
        self,
        c_z: int,
        n: int,
    ) -> None:
        super().__init__()
        self.layer_norm = LayerNorm(c_z)
        self.linear_1 = Linear(c_z, n * c_z, bias=True, init="relu")
        self.linear_2 = Linear(n * c_z, c_z, bias=True, init="final")

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Pair Transition forward pass.

        Args:
            z: [batch, N_res, N_res, c_z] pair representation
            mask: [batch, N_res, N_res] pair mask

        Returns:
            z_update: [batch, N_res, N_res, c_z] pair representation update

        """
        # DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = z.new_ones(z.shape[-1])

        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            z = self._chunk(z, mask, chunk_size)
        else:
            z = self._transition(z, mask)

        return z

    def _transition(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.layer_norm(z)
        z = self.linear_1(z)
        z = torch.relu(z)
        z = self.linear_2(z) * mask
        return z

    @torch.jit.ignore
    def _chunk(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"z": z, "mask": mask},
            chunk_size=chunk_size,
            num_batch_dims=len(z.shape[:-2]),
        )
