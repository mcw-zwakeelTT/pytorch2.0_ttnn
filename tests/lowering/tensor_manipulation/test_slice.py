import torch
import torch_ttnn
import unittest
import ttnn
import tt_lib

from tests.utils import check_with_pcc


class SliceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, dim, start, end):
        return torch.ops.aten.slice.Tensor(input, dim, start, end)

    def input_shapes(self):
        return [(64, 96), 1, 32, 63]

class TestModules(unittest.TestCase):
    def setUp(self):
        # Open device 0
        self.device: ttnn.Device = ttnn.open_device(device_id=0)
        # For AutoFormat
        tt_lib.device.SetDefaultDevice(self.device)

    def tearDown(self):
        # Close the device
        ttnn.close_device(self.device)

    def test_slice(self):
        m = SliceModule()
        input_shapes = m.input_shapes()
        input = torch.rand(input_shapes[0], dtype=torch.bfloat16)
        dim = input_shapes[1]
        start = input_shapes[2]
        end = input_shapes[3]
        result_before = m.forward(input, dim, start, end)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(input, dim, start, end)
        option._out_fx_graphs[0].print_tabular()
        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)

        # self.assertTrue(nodes[4].target == ttnn.slice)
        # self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        # self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        # self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        # self.assertTrue(nodes[5].target == ttnn.from_device)
        # self.assertTrue(nodes[6].target == ttnn.to_layout)
        # self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after))

if __name__ == "__main__":
    unittest.main()
