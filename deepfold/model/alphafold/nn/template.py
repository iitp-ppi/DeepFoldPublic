# Copyright 2023 DeepFold Team
# Copyright 2021 DeepMind Technologies Limited

from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn

from deepfold.model.alphafold.nn.dropout import DropoutColumnwise, DropoutRowwise
from deepfold.model.alphafold.nn.primitives import Attention, LayerNorm, Linear
from deepfold.model.alphafold.nn.transitions import PairTransition
from deepfold.model.alphafold.nn.triangular_attention import TriangleAttentionEndingNode, TriangleAttentionStartingNode
from deepfold.model.alphafold.nn.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from deepfold.model.alphafold.utils.feats import build_template_angle_feat, build_template_pair_feat
from deepfold.utils.chunk_utils import chunk_layer
from deepfold.utils.tensor_utils import flatten_final_dims, permute_final_dims, tensor_tree_map


class TemplatePointwiseAttention(nn.Module):
    """
    Implements Algorithm 17
    """

    def __init__(self, c_t: int, c_z: int, c_hidden: int, num_heads: int, inf: float) -> None:
        super().__init__()

        self.c_t = c_t
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.inf = inf

        self.mha = Attention(self.c_z, self.c_t, self.c_t, self.c_hidden, self.num_heads, gating=False)

    def _chunk(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        pass
