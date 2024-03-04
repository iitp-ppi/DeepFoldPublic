from typing import Optional

import torch
import torch.nn as nn

from deepfold.model.v2.modules.layer_norm import LayerNorm
from deepfold.model.v2.modules.linear import Linear
from deepfold.utils.chunk_utils import chunk_layer


class MSATransition(nn.Module):
    """MSA Transition module.

    Supplementary '1.6.3 MSA transition': Algorithm 9.

    Args:
        c_m: MSA (or Extra MSA) representation dimension (channels).
        n: `c_m` multiplier to obtain hidden dimension (channels).

    """

    def __init__(
        self,
        c_m: int,
        n: int,
    ) -> None:
        super().__init__()
        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, n * c_m, bias=True, init="relu")
        self.linear_2 = Linear(n * c_m, c_m, bias=True, init="final")

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """MSA Transition forward pass.

        Args:
            m: [batch, N_seq, N_res, c_m] MSA representation
            mask: [batch, N_seq, N_res] MSA mask

        Returns:
            m_update: [batch, N_seq, N_res, c_m] MSA representation update

        """
        # DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = m.new_ones(m.shape[-1])

        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._transition(m, mask)

        return m

    def _transition(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = torch.relu(m)
        m = self.linear_2(m) * mask
        return m

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"m": m, "mask": mask},
            chunk_size=chunk_size,
            num_batch_dims=len(
                m.shape[:-2],
            ),
        )
