import torch

from deepfold.common import residue_constants as rc
from deepfold.losses.utils import sigmoid_cross_entropy
from deepfold.utils.tensor_utils import masked_mean


def experimentally_resolved_loss(
    logits: torch.Tensor,
    atom37_atom_exists: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    min_resolution: float,
    max_resolution: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Predicts if an atom is experimentally resolved in a high-res structure.

    Jumper et al. (2021) Suppl. Sec. 1.9.10 '"Experimentally resolved" prediction'

    Args:
        logits: logits of shape [*, N_res, 37]. Log probability that an atom is resolved
        in atom37 representation, can be converted to probability by applying sigmoid.
        atom37_atom_exists: labels of shape [*, N_res, 37]
        all_atom_mask: mask of shape [*, N_res, 37]
        resolution: resolution of each example of shape [*]

    NOTE:
        This loss is used during fine-tuning on high-resolution X-ray crystals
        and cryo-EM structures resolution better than 0.3 nm.
        NMR and distillation examples have zero resolution.
    """

    errors = sigmoid_cross_entropy(logits=logits, labels=all_atom_mask)
    loss = torch.sum(errors * atom37_atom_exists, dim=-1)
    loss = loss / (eps + torch.sum(atom37_atom_exists, dim=(-1, -2)).view(-1, 1))
    loss = torch.sum(loss, dim=-1)
    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))
    return loss


def repr_norm_loss(
    msa_norm: torch.Tensor,
    pair_norm: torch.Tensor,
    msa_mask: torch.Tensor,
    pseudo_beta_mask: torch.Tensor,
    eps=1e-5,
    tolerance=0.0,
) -> torch.Tensor:
    """Representation norm loss of Uni-Fold."""

    def norm_loss(x):
        max_norm = x.shape[-1] ** 0.5
        norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
        error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
        return error

    pair_norm_error = norm_loss(pair_norm.float())
    msa_norm_error = norm_loss(msa_norm.float())
    pair_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

    pair_norm_loss = masked_mean(pair_mask.float(), pair_norm_error, dim=(-1, -2))
    msa_norm_loss = masked_mean(msa_mask.float(), msa_norm_error, dim=(-1, -2))

    loss = pair_norm_loss + msa_norm_loss

    return loss


def get_asym_mask(asym_id):
    """Get the mask for each asym_id. [*, NR] -> [*, NC, NR]"""
    # this func presumes that valid asym_id ranges [1, NC] and is dense.
    asym_type = torch.arange(1, torch.amax(asym_id) + 1, device=asym_id.device)  # [NC]
    return (asym_id[..., None, :] == asym_type[:, None]).float()


def chain_centre_mass_loss(
    pred_atom_positions: torch.Tensor,
    true_atom_positions: torch.Tensor,
    atom_mask: torch.Tensor,
    asym_id: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:

    ca_pos = rc.atom_order["CA"]
    pred_atom_positions = pred_atom_positions[..., ca_pos, :].float()  # [B, NR, 3]
    true_atom_positions = true_atom_positions[..., ca_pos, :].float()  # [B, NR, 3]
    atom_mask = atom_mask[..., ca_pos].bool()  # [B, NR]
    assert len(pred_atom_positions.shape) == 3

    asym_mask = get_asym_mask(asym_id) * atom_mask[..., None, :]  # [B, NC, NR]
    asym_exists = torch.any(asym_mask, dim=-1).float()  # [B, NC]

    def get_asym_centres(pos):
        pos = pos[..., None, :, :] * asym_mask[..., :, :, None]  # [B, NC, NR, 3]
        return torch.sum(pos, dim=-2) / (torch.sum(asym_mask, dim=-1)[..., None] + eps)

    pred_centres = get_asym_centres(pred_atom_positions)  # [B, NC, 3]
    true_centres = get_asym_centres(true_atom_positions)  # [B, NC, 3]

    def get_dist(p1: torch.Tensor, p2: torch.Tensor):
        return torch.sqrt((p1[..., :, None, :] - p2[..., None, :, :]).square().sum(-1) + eps)

    pred_centres2 = pred_centres
    true_centres2 = true_centres
    pred_dists = get_dist(pred_centres, pred_centres2)  # [B, NC, NC]
    true_dists = get_dist(true_centres, true_centres2)  # [B, NC, NC]
    losses = (pred_dists - true_dists + 4).clamp(max=0).square() * 0.0025
    loss_mask = asym_exists[..., :, None] * asym_exists[..., None, :]  # [B, NC, NC]

    loss = masked_mean(loss_mask, losses, dim=(-1, -2))

    return loss
