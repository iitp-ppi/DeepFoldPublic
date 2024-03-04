# DeepFold Team

import torch


def is_fp16_enabled() -> bool:
    return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16
