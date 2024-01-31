import unittest

import torch
import torch.distributed as dist

from deepfold.testing import MultiProcessTestCase

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 6000
INIT_METHOD = f"tcp://{MASTER_ADDR}:{MASTER_PORT}"


class TestNccl(MultiProcessTestCase):
    def _create_process_group_nccl(self):
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            init_method=INIT_METHOD,
        )

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 2

    def test_empty_tensors(self):
        self._create_process_group_nccl()

        device = torch.device("cuda", self.rank)

        x = torch.zeros(32, device=device)

        self.assertEqual((x == 0).all(), True)

    def test_all_reduce(self):
        self._create_process_group_nccl()

        device = torch.device("cuda", self.rank)

        x = torch.ones(32, device=device) * 2 * self.rank
        dist.all_reduce(x)
        dist.barrier()

        self.assertEqual((x == 2 * sum(range(self.world_size))).all(), True)


if __name__ == "__main__":
    unittest.main()
