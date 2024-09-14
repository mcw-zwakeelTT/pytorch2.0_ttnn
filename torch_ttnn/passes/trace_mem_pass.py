import torch
import ttnn
import pandas as pd
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch_ttnn.passes.lowering.add_data_move_pass import is_tt_compute
from torch_ttnn.utils import TtnnRunModeNormal, TtnnRunModeNoDispatch, TtnnDevice

from torch_ttnn.mem_utils import *
import json
import traceback

import faulthandler
faulthandler.enable()


def extract_peak_L1_memory_usage(trace):
    total_cb = 0
    total_buffer = 0
    peak_memory_usage = 0
    current_op = []

    for i in range(len(trace)):
        v = trace[i]

        if v["node_type"] == "function_start":
            if not current_op:
                while i + 1 < len(trace):
                    i += 1
                    inner_v = trace[i]
                    if inner_v["node_type"] == "buffer" and inner_v["params"]["type"] == "L1":
                        total_buffer += int(inner_v["params"]["size"])
                    elif inner_v["node_type"] == "tensor":
                        continue
                    else:
                        break
                i -= 1  # adjust for loop increment
            current_op.append(v["params"]["name"])
        elif v["node_type"] == "circular_buffer_allocate":
            total_cb += int(v["params"]["size"])
        elif v["node_type"] == "circular_buffer_deallocate_all":
            total_cb = 0
        elif v["node_type"] == "buffer_allocate" and v["params"]["type"] == "L1":
            total_buffer += int(v["params"]["size"])
        elif v["node_type"] == "buffer_allocate":
            connection = v["connections"][0]
            buffer = trace[connection]
            if buffer["params"]["type"] == "L1":
                total_buffer -= int(buffer["params"]["size"])
        elif v["node_type"] == "function_end":
            current_op.pop()
        peak_memory_usage = max(peak_memory_usage, total_cb + total_buffer)

    # if peak_memory_usage > mem_state.peak_sram_usage:
    #     mem_state.peak_sram_usage = peak_memory_usage
    # print(f"peak memory usage at current timestep: {peak_memory_usage}")
    # print(f"peak sram usage for the model: {mem_state.peak_sram_usage}")
    return peak_memory_usage


def process_allocations(graph):
    df = pd.DataFrame(columns=["current_op", "event", "total_cb", "total_buffer", "info"])

    cur_op = []
    total_cb = 0
    total_buffer = 0
    tensors = set()
    i = 1  # lets skip initial node
    while i < len(graph):
        params = ""
        v = graph[i]
        params = v["params"]
        print(v, len(df))
        i += 1
        if v["node_type"] == "function_start":
            if len(cur_op) == 0:
                # entring first op, lets get all input tensors
                while i < len(graph):
                    print(graph[i], len(df))
                    if graph[i]["node_type"] == "buffer":
                        total_buffer += int(graph[i]["params"]["size"])
                        i += 1
                    elif graph[i]["node_type"] == "tensor":
                        i += 1
                    else:
                        break
            name = v["params"]["name"]
            if name == "ttnn::prim::old_infra_device_operation":
                name = "ttnn::prim::old_infra_op"
            cur_op.append(name)
        if v["node_type"] == "circular_buffer_allocate":
            total_cb += int(v["params"]["size"])
        if v["node_type"] == "circular_buffer_deallocate_all":
            total_cb = 0
        if v["node_type"] == "buffer_allocate":
            total_buffer += int(v["params"]["size"])
        if v["node_type"] == "function_end":
            cur_op.pop()
            # continue
        if v["node_type"] == "tensor":
            continue
        if v["node_type"] == "buffer_deallocate":
            total_buffer -= int(graph[v["connections"][0]]["params"]["size"])
        if v["node_type"] == "buffer":
            continue
        if len(cur_op) > 0:
            data = {
                "current_op": cur_op[-1],
                "event": v["node_type"],
                "total_cb": total_cb,
                "total_buffer": total_buffer,
                "info": params,
            }
            df.loc[len(df)] = data
    for data in df:
        print(df)
    return df


def logger(trace):
    trace = json.dumps(trace)
    log_file = f"metrics/memory_footprint.txt"
    with open(log_file, "a") as f:
        f.write(trace)


def check_sram_overflow(memory_state: MemoryState):
    if memory_state.peak_sram_usage > SRAM_LIMIT:
        memory_state.fits_in_memory = False
    else:
        memory_state.fits_in_memory = True


def get_input_tensors_meta(graph):
    ttnn_operations = []  # To store details of TT-NN operations
    tensors = {}  # To map tensor node_id to tensor shape

    # First, gather all tensors and their shapes
    for node in graph:
        if node['node_type'] == 'tensor':
            tensor_id = node['counter']  # The unique identifier for the tensor
            shape = node['params'].get('shape', None)  # Tensor shape
            tensors[tensor_id] = shape

    # Now, look for TT-NN operations and gather their input tensor details
    for node in graph:
        # Check if the node is a function_start and its name contains 'ttnn::'
        if node['node_type'] == 'function_start' and 'name' in node['params']:
            operation_name = node['params']['name']
            
            # Check if the operation is a TT-NN operation
            if operation_name.startswith('ttnn::'):
                # Extract number of input tensors
                inputs = int(node['params'].get('inputs', 0))

                # Collect connected tensors based on the connections to this node
                input_tensor_sizes = []
                for conn in node['connections']:
                    # Check if the connected node is a tensor and has a shape
                    if conn in tensors:
                        input_tensor_sizes.append(tensors[conn])

                # Add operation details along with the input tensor sizes
                ttnn_operations.append({
                    'operation': operation_name,
                    'inputs': inputs,
                    'input_tensor_sizes': input_tensor_sizes,
                    'node_id': node['counter'],  # The unique identifier of the node
                })
    
    return ttnn_operations


class TraceMemoryPass(PassBase):
    def __init__(self, device):
        self.device = device
        self.op_registry = OpRegistry()

    def trace_memory(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # mode = TtnnRunModeNoDispatch()
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)

        # Pytorch by default enables fake tensor mode in fx graph parsing.
        # So actual tensors are not created, rather its a FakeTensor.
        # So fake tensor mode has to be explicitly disabled.
        from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
        with maybe_disable_fake_tensor_mode():

            nodes = list(gm.graph.nodes)
            for node in nodes:
                if is_tt_compute(node):
                    print(f"Tracing {node.name}...")

                    # If reshape on host, then skip
                    if (
                        node.target == ttnn.reshape and
                        (node.args[0].target != ttnn.from_torch or
                        "device" not in node.kwargs or
                        not isinstance(node.kwargs["device"], TtnnDevice)) or
                        node.target == ttnn.full
                        ):
                        continue

                    inputs = []
                    for input_node in node.all_input_nodes:
                        tensor_shape, _ = self.op_registry.get_tensor_shape_and_dtype(input_node)
                        torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
                        ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
                        inputs.append(ttnn_tensor)

                    # Assumption: Tensors are first in the order of inputs
                    # Inputs which are not tensors
                    for arg in node.args:
                        if isinstance(arg, tuple) or isinstance(arg, list):
                            inputs.append(arg)

                    output_tensor = node.target(*inputs)

            torch_output_tensor = ttnn.to_torch(output_tensor)

            captured_graph = ttnn.graph.end_graph_capture()

        tensors_meta = get_input_tensors_meta(captured_graph)

        print(f"\n\nTensor meta:")
        print(json.dumps(tensors_meta, indent=4))
        print(f"\n\nTrace json:")
        print(json.dumps(captured_graph, indent=4))
        print(f"\nPeak SRAM usage for the model: {extract_peak_L1_memory_usage(captured_graph)}")

        return gm

    def call(self, gm: torch.fx.GraphModule):
        gm = self.trace_memory(gm)
        # import traceback
        # try:
        #     gm.recompile()
        # except Exception as e:
        #     print(traceback.format_exc())
        return PassResult(gm, True)
