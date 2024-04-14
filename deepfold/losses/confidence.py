import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold.common import residue_constants as rc
from deepfold.losses.geometry import compute_lddt
from deepfold.losses.utils import sigmoid_cross_entropy, softmax_cross_entropy
from deepfold.utils.tensor_utils import batched_gather, masked_mean


def plddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    cutoff: float = 15.0,
    num_bins: int = 50,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-10,
) -> torch.Tensor:

    ca_pos = rc.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :].float()
    all_atom_positions = all_atom_positions[..., ca_pos, :].float()
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)].float()  # Keep dim

    lddt = compute_lddt(all_atom_pred_pos, all_atom_positions, all_atom_mask, cutoff=cutoff, eps=eps).detach()

    bin_index = torch.floor(lddt * num_bins).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
    lddt_ca_one_hot = F.one_hot(bin_index, num_classes=num_bins)

    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)

    loss = masked_mean(all_atom_mask, errors, dim=-1, eps=eps)
    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))

    return loss
