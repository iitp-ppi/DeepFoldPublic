import unittest

import torch

from deepfold.distributed.tensor import DTensor, Mesh, Replicate
from deepfold.distributed.tensor.utils import _create_1d_device_mesh
from deepfold.testing.common_dtensor import DTensorTestBase, MLPModule, with_comms


class TensorParallelAPITests(DTensorTestBase):
    @property
    def world_size(self):
        gpu_num = torch.cuda.device_count()
        return gpu_num if gpu_num % 2 == 0 and gpu_num > 4 else 4

    @with_comms
    def test_create_1d_device_mesh(self):
        dim_one_size = 2
        mesh_shape = (
            torch.arange(self.world_size)
            .reshape(
                self.world_size // dim_one_size,
                dim_one_size,
            )
            .to(torch.int)
        )
        mesh = Mesh(self.device_type, mesh_shape)
        # When 1D dim is 1.
        one_dimention_mesh_shape = mesh_shape[self.rank // dim_one_size, :]
        pg = mesh.get_dim_groups()[1]
        new_mesh = _create_1d_device_mesh(mesh, 1)
        expected_mesh = one_dimention_mesh_shape

        self.assertTrue(torch.all(new_mesh.mesh == expected_mesh))
        self.assertEqual(new_mesh.device_type, self.device_type)
        self.assertEqual(new_mesh.get_dim_groups(), [pg])
        # When 1D dim is 0.
        one_dimention_mesh_shape = mesh_shape[:, self.rank % dim_one_size]
        pg = mesh.get_dim_groups()[0]
        new_mesh = _create_1d_device_mesh(mesh, 0)
        expected_mesh = one_dimention_mesh_shape
        self.assertTrue(torch.all(new_mesh.mesh == expected_mesh))
        self.assertEqual(new_mesh.device_type, self.device_type)
        self.assertEqual(new_mesh.get_dim_groups(), [pg])

    @with_comms
    def test_create_1d_device_mesh_error(self):
        mesh = Mesh(self.device_type, torch.arange(self.world_size))
        with self.assertRaisesRegex(
            AssertionError,
            "Expect tp_mesh_dim within range \\[-1, 1\\), but found 3.",
        ):
            _create_1d_device_mesh(mesh, 3)

    def _compare_params(
        self,
        local_module,
        dist_module,
        rank0_only,
        skip_rowwise_bias=False,
        compare_grad=False,
    ):
        replicate = [Replicate()]
        for name, param in local_module.named_parameters():
            dist_param = dist_module.get_parameter(name)
            param = param.grad if compare_grad else param
            dist_param = dist_param.grad if compare_grad else dist_param
            if (
                (not rank0_only)
                or (self.rank == 0)
                or (name not in ["net2.bias"] and not skip_rowwise_bias or name not in ["bias", "net2.bias"])
            ):
                self.assertEqual(
                    param,
                    dist_param.redistribute(device_mesh=dist_param.device_mesh, placements=replicate).to_local(),
                    f"{name} not equal between dist and non-dist",
                )

    def _compare_module(self, local_module, dist_module, inp_size, rank0_only=True, rowwise=False):
        LR = 0.25  # the learning rate we use for testing
        local_optim = torch.optim.SGD(local_module.parameters(), lr=LR)
        dist_optim = torch.optim.SGD(dist_module.parameters(), lr=LR)
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        self._compare_params(local_module, dist_module, rank0_only)

        # check forward correctness
        local_output = local_module(inp)
        inp = inp.chunk(self.world_size, dim=-1)[self.rank] if rowwise else inp
        dist_output = dist_module(inp)
        dist_output = (
            dist_output.redistribute(dist_output.device_mesh, [Replicate()]).to_local()
            if isinstance(dist_output, DTensor)
            else dist_output
        )
        self.assertEqual(local_output, dist_output)

        local_output.sum().backward()
        dist_output.sum().backward()

        # check backward and ensure gradients are same
        self._compare_params(local_module, dist_module, rank0_only, rowwise, True)

        local_optim.step()
        dist_optim.step()
        self._compare_params(local_module, dist_module, rank0_only, rowwise)


if __name__ == "__main__":
    unittest.main()
