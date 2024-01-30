import logging
import os

import torch
import torch.distributed as dist

import deepfold.core.parallel_state as ps

logger = logging.getLogger(__name__)


class Distributed:
    world_size: int = torch.cuda.device_count()
    rank: int = int(os.environ["LOCAL_RANK"])

    @staticmethod
    def initialize_distributed():
        if not dist.is_initialized():
            logger.debug(
                f"Initializing torch.distributed with rank: {Distributed.rank}, world_size: {Distributed.world_size}"
            )
            torch.cuda.set_device(Distributed.rank % torch.cuda.device_count())
            init_method = "tcp://"
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", "6000")
            init_method += f"{master_ip}:{master_port}"
            dist.init_process_group(
                backend="nccl",
                world_size=Distributed.world_size,
                rank=Distributed.rank,
                init_method=init_method,
            )

    @staticmethod
    def destroy_model_parallel():
        ps.destroy_model_parallel()
        dist.barrier()

    @staticmethod
    def initialize_distributed(model_parallel_size: int = 1):
        ps.destroy_model_parallel()
        Distributed.initialize_distributed()
        ps.initialize_model_parallel(model_parallel_size)
