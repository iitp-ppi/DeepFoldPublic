import unittest

import torch
import torch.distributed as dist

from deepfold.core.model_parallel import mappings
from deepfold.testing import MultiProcessTestCase
from tests.utils import Distributed

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 6000
INIT_METHOD = f"tcp://{MASTER_ADDR}:{MASTER_PORT}"
NGPUS = torch.cuda.device_count()


class TestModelParallelMappings(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 2

    # @unittest.skip("")
    def test_CopyToModelParallelRegion(self):
        Distributed.initialize_model_parallel(self.world_size, self.rank, NGPUS)
        x1 = torch.ones((1)).cuda() * self.rank
        x1.requires_grad_()
        y1 = mappings.copy_to_model_parallel_reigon(x1)
        self.assertTrue(torch.equal(y1, x1))
        y1.sum().backward()
        g1 = torch.ones((1)).cuda() * self.world_size
        self.assertTrue(torch.equal(x1.grad, g1))
        Distributed.destroy_model_parallel()

    # @unittest.skip("")
    def test_ReduceFromModelParallelRegion(self):
        Distributed.initialize_model_parallel(self.world_size, self.rank, NGPUS)
        x1 = torch.ones((1)).cuda() * self.rank
        x1.requires_grad_()
        y1 = mappings.reduce_from_model_parallel_region(x1)
        y2 = torch.ones((1)).cuda() * sum(range(self.world_size))
        self.assertTrue(torch.equal(y1, y2))
        y1.sum().backward()
        self.assertTrue(torch.equal(torch.ones_like(x1), x1.grad))
        Distributed.destroy_model_parallel()

    # @unittest.skip("")
    def test_ScatterToModelParallelRegion(self):
        Distributed.initialize_model_parallel(self.world_size, self.rank, NGPUS)
        x1 = torch.arange(8 * self.world_size, dtype=torch.float32).reshape(8, -1).cuda()
        x1.requires_grad_()
        y2 = mappings.scatter_to_model_parallel_region(x1)
        self.assertTrue(torch.equal(y2, x1.chunk(self.world_size, dim=-1)[self.rank]))
        y2.sum().backward()
        self.assertTrue(torch.equal(x1.grad, torch.ones_like(x1)))
        Distributed.destroy_model_parallel()


if __name__ == "__main__":
    unittest.main()
