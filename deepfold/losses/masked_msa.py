import torch
import torch.nn.functional as F

from deepfold.losses.utils import softmax_cross_entropy


def masked_msa_loss(
    logits: torch.Tensor,
    true_msa: torch.Tensor,
    bert_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Computes BERT-style masked MSA loss.

    Supplementary '1.9.9 Masked MSA prediction'.

    Args:
        logits:    [*, N_seq, N_res, 23] predicted residue distribution
        true_msa:  [*, N_seq, N_res] true MSA
        bert_mask: [*, N_seq, N_res] MSA mask

    Returns:
        Masked MSA loss

    """
    errors = softmax_cross_entropy(logits=logits, labels=F.one_hot(true_msa, num_classes=23))

    # FP16-friendly averaging.
    loss = errors * bert_mask
    loss = torch.sum(loss, dim=-1)
    scale = 0.5
    denom = eps + torch.sum(scale * bert_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    return loss
