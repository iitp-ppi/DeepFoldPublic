import torch

from deepfold.losses.utils import sigmoid_cross_entropy


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

    atom37_atom_exists = atom37_atom_exists.float()
    all_atom_mask = all_atom_mask.float()
    errors = sigmoid_cross_entropy(logits.float(), all_atom_mask)
    loss = torch.sum(errors * atom37_atom_exists, dim=-1)
    denom = torch.sum(atom37_atom_exists, dim=(-1, -2)).unsqueeze(-1)

    loss = loss / (eps + denom)
    loss = torch.sum(loss, dim=-1)
    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))

    return loss
