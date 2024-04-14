import torch


def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Softmax cross entropy."""
    loss = -1 * torch.sum(labels * torch.nn.functional.log_softmax(logits.float(), dim=-1), dim=-1)
    return loss


def sigmoid_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    log_p = torch.nn.functional.logsigmoid(logits)
    log_not_p = torch.nn.functional.logsigmoid(-logits)  # More numerically stable.
    loss = -labels * log_p - (1 - labels) * log_not_p
    return loss


def calculate_bin_centers(boundaries: torch.Tensor) -> torch.Tensor:
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat([bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0)
    return bin_centers
