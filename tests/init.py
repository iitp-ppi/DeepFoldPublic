import os

from deepfold.distributed.legacy import init_distributed, is_initialized


def init_parallel(local_rank: int, world_size: int, port: int = 16141, random_seed: int = 0):
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    if not is_initialized():
        init_distributed(tensor_model_parallel_size=world_size, random_seed=random_seed)
