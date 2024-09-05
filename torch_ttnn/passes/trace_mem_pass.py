import torch
import ttnn
import pandas as pd
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch_ttnn.passes.lowering.add_data_move_pass import is_tt_compute
from torch_ttnn.utils import TtnnRunModeNormal, TtnnRunModeNoDispatch

from torch_ttnn.mem_utils import MemoryState, SRAM_LIMIT
import json


def extract_peak_L1_memory_usage(trace, mem_state: MemoryState):
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

    if peak_memory_usage > mem_state.peak_sram_usage:
        mem_state.peak_sram_usage = peak_memory_usage
    print(f"peak memory usage at current timestep: {peak_memory_usage}")
    print(f"peak sram usage for the model: {mem_state.peak_sram_usage}")
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


class TraceMemoryPass(PassBase):
    def __init__(self, memory_state: MemoryState):
        self.memory_state = memory_state

    def trace_memory(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        mode = TtnnRunModeNormal()
        nodes = list(gm.graph.nodes)
        for node in nodes:
            if is_tt_compute(node):
                with gm.graph.inserting_before(node):
                    gm.graph.call_function(ttnn.graph.begin_graph_capture, args=(mode,))
                with gm.graph.inserting_after(node):
                    res = gm.graph.call_function(ttnn.graph.end_graph_capture, args=())
                with gm.graph.inserting_after(res):
                    peak_sram = gm.graph.call_function(extract_peak_L1_memory_usage, args=(res, self.memory_state))
                with gm.graph.inserting_after(peak_sram):
                    gm.graph.call_function(logger, args=(res,))
        with gm.graph.inserting_after():
            gm.graph.call_function(check_sram_overflow, args=(self.memory_state,))
        return gm

    def call(self, gm: torch.fx.GraphModule):
        gm = self.trace_memory(gm)
        print(gm.code)
        for node in list(gm.graph.nodes):
            print(node)
        # import traceback
        # try:
        #     gm.recompile()
        # except Exception as e:
        #     print(traceback.format_exc())
        return PassResult(gm, True)
