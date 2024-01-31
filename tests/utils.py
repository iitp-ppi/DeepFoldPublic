import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

import deepfold.core.parallel_state as ps

logger = logging.getLogger(__name__)


class Distributed:
    @staticmethod
    def initialize_distributed(world_size: int, rank: int, ngpus: int):
        if not dist.is_initialized():
            logger.debug(f"Initializing torch.distributed with rank: {rank}, world_size: {world_size}")
            torch.cuda.set_device(rank % ngpus)
            init_method = "tcp://"
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", "6000")
            init_method += f"{master_ip}:{master_port}"

            dist.init_process_group(
                backend="nccl",
                world_size=world_size,
                rank=rank,
                init_method=init_method,
            )

    @staticmethod
    def destroy_model_parallel():
        ps.destroy_model_parallel()
        dist.barrier()

    @staticmethod
    def initialize_model_parallel(model_parallel_size: int, rank: int, ngpus: int):
        ps.destroy_model_parallel()
        Distributed.initialize_distributed(model_parallel_size, rank, ngpus)
        ps.initialize_model_parallel(model_parallel_size, rank)
