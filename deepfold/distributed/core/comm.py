import logging

import torch
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.distributed_c10d import all_gather_into_tensor as _all_gather_into_tensor

logger = logging.getLogger(__name__)


def all_gather_tensor(
    self: torch.Tensor,
    gather_dim: int,
    group: ProcessGroup,
) -> torch.Tensor:
    assert self.is_contiguous()
    group_size = group.size()
    output_size = (group_size, *self.size())
    tensor = torch.empty(output_size, dtype=self.dtype, device=self.device)
    _all_gather_into_tensor(tensor, self, group=group, async_op=False)
    if gather_dim != 0:
        tensor = torch.cat(torch.chunk(tensor, group_size, dim=0), dim=gather_dim)
    return tensor
