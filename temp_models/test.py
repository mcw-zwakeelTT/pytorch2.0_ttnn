import torch
import ttnn

device_id = 0
device = ttnn.open_device(device_id=device_id)

ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)

inputs = []     
breakpoint()    
from torch._subclasses.fake_tensor import FakeTensorMode
fake_mode = FakeTensorMode()
with fake_mode:
    from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
    with maybe_disable_fake_tensor_mode(): 
        t1 = torch.rand((2, 4), dtype=torch.float32)
        t1 = ttnn.from_torch(t1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
inputs.append(t1)
t2 = torch.rand((2, 4), dtype=torch.float32)
t2 = ttnn.from_torch(t2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
inputs.append(t2)

output_tensor = ttnn.add(*inputs)
torch_output_tensor = ttnn.to_torch(output_tensor)

captured_graph = ttnn.graph.end_graph_capture()
ttnn.close_device(device)