import logging

from torch.distributed.distributed_c10d import ProcessGroup, _get_default_group

logger = logging.getLogger(__name__)


def get_default_group() -> ProcessGroup:
    return _get_default_group()
