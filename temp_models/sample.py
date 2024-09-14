import torch
import ttnn
import torch_ttnn
import torch.nn.functional as F
import json

class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        # return z
        return F.relu(z)


class ReluModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x)
      

def torch_to_ttnn_op():
    input_shapes = [(2, 4), (2, 4)]
    inputs = [torch.randint(1, 5, shape).type(torch.bfloat16) for shape in input_shapes]
    m = Module()
    # result_before = m.forward(*inputs)
    option = torch_ttnn.TorchTtnnOption(device=device, gen_graphviz=False, run_tracer_pass=True)
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs)

def ttnn_direct(device):
    mode = ttnn.graph.RunMode.NO_DISPATCH
    ttnn.graph.begin_graph_capture(mode)
    
    torch_input_tensor = torch.rand(2, 4, dtype=torch.float32)
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # output_tensor = ttnn.exp(input_tensor)
    output_tensor = ttnn.full([2,4], 5.0)
    torch_output_tensor = ttnn.to_torch(output_tensor)

    res = ttnn.graph.end_graph_capture()
    print(json.dumps(res, indent=4))

if __name__ == "__main__":
    device_id = 0
    device = ttnn.open_device(device_id=device_id)
    
    torch_to_ttnn_op()
    # ttnn_direct(device)

    ttnn.close_device(device)