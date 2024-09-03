import torch
import ttnn
import pandas as pd
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch_ttnn.passes.lowering.add_data_move_pass import is_tt_compute
from torch_ttnn.utils import TtnnRunModeNormal, TtnnRunModeNoDispatch


def extract_peak_L1_memory_usage(trace):
    total_cb = 0
    total_buffer = 0
    peak_memory_usage = 0
    current_op = []

    for i in range(len(trace)):
        v = trace[i]

        if v["node_type"] == 'function_start':
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
    print(f"peak memory usage: {peak_memory_usage}")
    return peak_memory_usage



def process_allocations(graph):
    df = pd.DataFrame(columns=['current_op', 'event', 'total_cb', 'total_buffer', 'info'])
    
    cur_op = []
    total_cb = 0
    total_buffer = 0
    tensors = set()
    i = 1 # lets skip initial node
    while i < len(graph):
        params = ''
        v = graph[i]
        params = v["params"]
        print(v, len(df))
        i += 1
        if v["node_type"] == 'function_start':
            if len(cur_op) == 0:
                #entring first op, lets get all input tensors
                while i < len(graph):
                    print(graph[i], len(df))
                    if graph[i]["node_type"] == 'buffer':
                        total_buffer += int(graph[i]["params"]['size'])
                        i += 1
                    elif graph[i]["node_type"] == 'tensor':
                        i += 1
                    else:
                        break
            name = v["params"]['name']
            if name == "ttnn::prim::old_infra_device_operation":
                name = "ttnn::prim::old_infra_op"
            cur_op.append(name)
        if v["node_type"] == 'circular_buffer_allocate':
            total_cb += int(v["params"]['size'])
        if v["node_type"] == 'circular_buffer_deallocate_all':
            total_cb = 0
        if v["node_type"] == 'buffer_allocate':
            total_buffer += int(v["params"]['size'])
        if v["node_type"] == 'function_end':
            cur_op.pop()
            #continue
        if v["node_type"] == 'tensor':
            continue
        if v["node_type"] == 'buffer_deallocate':
            total_buffer -= int(graph[v["connections"][0]]["params"]['size'])
        if v["node_type"] == 'buffer':
            continue
        if len(cur_op) > 0:
            data =  {'current_op': cur_op[-1], 'event' : v["node_type"], 'total_cb': total_cb, 'total_buffer': total_buffer, 'info' : params}
            df.loc[len(df)] = data
    for data in df:
        print(df)
    return df



class TraceMemoryPass(PassBase):

    def trace_memory(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        print("In trace memory pass")
        mode = TtnnRunModeNormal()
        nodes = list(gm.graph.nodes)
        for node in nodes:
            if is_tt_compute(node):
                with gm.graph.inserting_before(node):
                    gm.graph.call_function(ttnn.graph.begin_graph_capture, args=(mode,))
                with gm.graph.inserting_after(node):
                    res = gm.graph.call_function(ttnn.graph.end_graph_capture, args=())
                with gm.graph.inserting_after(res):
                    gm.graph.call_function(extract_peak_L1_memory_usage, args=(res,))
        return gm

    def call(self, gm: torch.fx.GraphModule):
        gm = self.trace_memory(gm)
        print(gm.code)
        for node in list(gm.graph.nodes):
            print(node)
        return PassResult(gm, True)
