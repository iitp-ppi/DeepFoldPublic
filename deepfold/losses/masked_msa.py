import torch
import torch.nn.functional as F

from deepfold.losses.utils import softmax_cross_entropy
from deepfold.utils.tensor_utils import masked_mean


def masked_msa_loss(logits: torch.Tensor, true_msa: torch.Tensor, bert_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    bert_mask = bert_mask.float()
    errors = softmax_cross_entropy(logits.float(), F.one_hot(true_msa.long(), num_classes=logits.shape[-1]))
    loss = masked_mean(bert_mask, errors, dim=(-1, -2), eps=eps)
    return loss
