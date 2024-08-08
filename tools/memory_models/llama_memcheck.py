import torch
import torch_ttnn
import ttnn

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":
    # Open device 0
    device: ttnn.Device = ttnn.open_device(device_id=0)

    # Download model from cloud
    model_name = "huggyllama/llama-7b"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", torch_dtype=torch.bfloat16
    )
    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    m.eval()

    # Set up sample input
    test_input = "This is a sample text from "
    inputs = tokenizer(test_input, return_tensors="pt")

    metrics_path = "llama"

    # Compile the model
    option = torch_ttnn.TorchTtnnOption(
        device=device, metrics_path=metrics_path
    )
    option.gen_graphviz = True
    m = torch.compile(m, backend=torch_ttnn.memory_backend, options=option)

    # Run inference with the compiled model
    with torch.no_grad():
        output_after = m(**inputs)

    # These are for plotting charts for later inspection
    from tools.memory_models.plot_chart import plot_bar_chart, plot_line_chart
    src_file = "./data/memory/memory_footprint.txt"
    bar_chart_file = "./tools/memory_models/assets/llama_bar_chart.png"
    line_chart_file = "./tools/memory_models/assets/llama_line_chart.png"
    plot_bar_chart(src_file, bar_chart_file)
    plot_line_chart(src_file, line_chart_file)    

    # Close the device
    ttnn.close_device(device)