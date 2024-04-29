import unittest

import torch
import torch.distributed as dist

from deepfold.distributed import model_parallel
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

    @unittest.skip("")
    def test_CopyToModelParallelRegion(self):
        Distributed.initialize_model_parallel(self.world_size, self.rank, NGPUS)
        x1 = torch.ones((1)).cuda() * self.rank
        x1.requires_grad_()
        y1 = model_parallel.copy_to_model_parallel_reigon(x1)
        self.assertTrue(torch.equal(y1, x1))
        y1.sum().backward()
        g1 = torch.ones((1)).cuda() * self.world_size
        self.assertTrue(torch.equal(x1.grad, g1))
        Distributed.destroy_model_parallel()

    @unittest.skip("")
    def test_ReduceFromModelParallelRegion(self):
        Distributed.initialize_model_parallel(self.world_size, self.rank, NGPUS)
        x1 = torch.ones((1)).cuda() * self.rank
        x1.requires_grad_()
        y1 = model_parallel.reduce_from_model_parallel_region(x1)
        y2 = torch.ones((1)).cuda() * sum(range(self.world_size))
        self.assertTrue(torch.equal(y1, y2))
        y1.sum().backward()
        self.assertTrue(torch.equal(torch.ones_like(x1), x1.grad))
        Distributed.destroy_model_parallel()

    @unittest.skip("")
    def test_ScatterToModelParallelRegion(self):
        Distributed.initialize_model_parallel(self.world_size, self.rank, NGPUS)
        x1 = torch.arange(8 * self.world_size, dtype=torch.float32).reshape(8, -1).cuda()
        x1.requires_grad_()
        y2 = model_parallel.scatter_to_model_parallel_region(x1)
        self.assertTrue(torch.equal(y2, x1.chunk(self.world_size, dim=-1)[self.rank]))
        y2.sum().backward()
        self.assertTrue(torch.equal(x1.grad, torch.ones_like(x1)))
        Distributed.destroy_model_parallel()

    @unittest.skip("")
    def test_GatherFromModelParallelRegion(self):
        Distributed.initialize_model_parallel(self.world_size, self.rank, NGPUS)
        x1 = torch.ones((16, self.world_size)).cuda()
        x1.requires_grad_()
        y1 = model_parallel.scatter_to_model_parallel_region(x1)
        self.assertTrue(torch.equal(y1, x1.new_ones((16, 1))))
        y1.sum().backward()
        self.assertTrue(torch.equal(x1.grad, torch.ones_like(x1)))
        Distributed.destroy_model_parallel()

    # @unittest.skip("")
    def test_TransposeOnModelParallelRegion(self):
        Distributed.initialize_model_parallel(self.world_size, self.rank, NGPUS)
        x1 = torch.ones(2, self.world_size, 3).cuda() * self.rank
        # x2: dim 1 chunked
        x2 = x1.chunk(self.world_size, dim=1)[self.rank]
        x2.requires_grad_()
        x3 = x2 * x2
        # y1: dim 0 chunked
        y1 = model_parallel.transpose_on_model_parallel_region(x3, 1, 0)
        y1.sum().backward()
        g2 = x2 * 2
        self.assertTrue(torch.equal(x2.grad, g2))
        Distributed.destroy_model_parallel()


if __name__ == "__main__":
    unittest.main()
