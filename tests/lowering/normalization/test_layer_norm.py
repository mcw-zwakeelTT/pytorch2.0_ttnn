import torch
import torch_ttnn
import pytest
import ttnn

from tests.utils import check_with_pcc


class LayerNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedding, weight, bias):
        return torch.nn.functional.layer_norm(
            embedding, normalized_shape=[embedding.shape[-1]], weight=weight, bias=bias
        )


class LayerNormWithOtherOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedding, weight, bias):
        layer_norm = torch.nn.functional.layer_norm(
            embedding, normalized_shape=[embedding.shape[-1]], weight=weight, bias=bias
        )
        return layer_norm + layer_norm


@pytest.mark.parametrize(
    "batch, sentence_length, embedding_dim",
    [(2, 32, 64)],
)
def test_layer_norm(device, batch, sentence_length, embedding_dim):
    m = LayerNormModule()
    input_shapes = [
        (batch, sentence_length, embedding_dim),
        (embedding_dim),
        (embedding_dim),
    ]
    inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
    result_before = m.forward(*inputs)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    assert [node.target for node in nodes].count(ttnn.layer_norm) == 1
    # Check inference result
    assert check_with_pcc(result_before, result_after, 0.9998)


@pytest.mark.parametrize(
    "batch, sentence_length, embedding_dim",
    [(2, 32, 64)],
)
def test_layer_norm_with_other_op(device, batch, sentence_length, embedding_dim):
    m = LayerNormWithOtherOpModule()
    input_shapes = [
        (batch, sentence_length, embedding_dim),
        (embedding_dim),
        (embedding_dim),
    ]
    inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
    result_before = m.forward(*inputs)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    assert [node.target for node in nodes].count(ttnn.layer_norm) == 1
    # Check inference result
    assert check_with_pcc(result_before, result_after, 0.9998)