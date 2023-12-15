import torch
import torch_ttnn
import unittest
from torch_ttnn import ttnn


class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + x

    def input_shapes(self):
        return [(4, 4)]


class MatmulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.matmul(x, x)

    def input_shapes(self):
        return [(4, 4)]


class TestModules(unittest.TestCase):
    def setUp(self):
        # Open device 0 and set it as torch_ttnn global variable
        self.device: ttnn.Device = ttnn.open(0)
        torch_ttnn.set_device(self.device)

    def tearDown(self):
        # Close the device
        ttnn.close(self.device)

    def test_add(self):
        m = AddModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        m = torch.compile(m, backend=torch_ttnn.backend)
        # TODO(yoco) Check the graph has be rewritten and contain ttnn ops
        result_after = m.forward(*inputs)
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_matmul(self):
        m = MatmulModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        m = torch.compile(m, backend=torch_ttnn.backend)
        # TODO(yoco) Check the graph has be rewritten and contain ttnn ops
        result_after = m.forward(*inputs)
        self.assertTrue(torch.allclose(result_before, result_after))


if __name__ == "__main__":
    unittest.main()