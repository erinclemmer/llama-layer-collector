# 🦙 Llama Layer Collector

![PyPI - Version](https://img.shields.io/pypi/v/llama-layer-collector)

A practical Python package for working with Llama-based models at the layer level. Designed to help developers and researchers load specific model components when working with large, sharded checkpoints.

## ✨ What It Does

- 🎯 Selective layer loading for more efficient resource usage
- 🚀 Streamlined handling of model checkpoints
- 💡 Useful for research, development, and memory-constrained environments

## 🛠️ Core Capabilities

Our package excels in four key areas:

### Precision Layer Control
Select which model components to load, from embedding layers to specific decoder blocks. This helps manage memory usage and processing requirements for your use case.

### Modular Architecture
Design your model processing pipeline by working with individual components. This approach enables focused testing, targeted optimization, and easier debugging of model behavior.

### Streamlined Computation
Use helper functions for embedding computation, layer-wise processing, and head operations to simplify working with model components.

## 🚀 Getting Started

### Installation

```bash
pip install llama-layer-collector
```

### Essential Components

The LlamaLayerCollector class serves as your central interface to the package's functionality. Here's what you need to know about its key parameters:

#### Required Parameters:
- `model_dir`: Path to your model directory containing shards and configuration
- `device`: Target device for tensor operations ("cpu" or "cuda")
- `dtype`: Desired numerical precision (default: torch.float16)

#### Optional Parameters:
- `cache_file`: Location for storing shard metadata
- `shard_pattern`: Custom regex for matching shard files
- `layer_prefix`: Prefix for identifying decoder layers
- Various layer name parameters for custom architectures

## 💻 Example Usage

Here's how you might use Llama Layer Collector in practice:

```python
from llama_layer_collector import LlamaLayerCollector
from llama_layer_collector.compute import compute_embedding, compute_layer, compute_head
from transformers import AutoTokenizer
import torch

# Initialize core components
collector = LlamaLayerCollector(
    model_dir="/path/to/model",
    cache_file="cache.json",
    device="cuda",
    dtype=torch.float16
)

# Set up tokenization
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")
input_text = "The quick brown fox"
input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']

# Load model components
embedding = collector.load_input_embedding()
norm = collector.load_norm()
head = collector.load_head()
layers = collector.load_layer_set(0, collector.num_layers - 1)

# Execute forward pass
state = compute_embedding(embedding, input_ids, collector.config)
for layer in layers:
    state.state = compute_layer(layer, state)

# Generate predictions
predictions = compute_head(head, norm(state.state), topk=1)
```

## Optimal Use Cases

### Resource-Constrained Environments
Perfect for scenarios where loading entire models is impractical or impossible. Load only the layers you need, when you need them.

### Model Development
Ideal for researchers and developers who need to:
- Analyze intermediate layer outputs
- Experiment with architectural modifications
- Implement custom layer combinations
- Debug model behavior at a granular level

### Production Optimization
Streamline production deployments by loading only essential model components, reducing memory footprint and improving resource utilization.

## ⚙️ Technical Details

### Shard Management
- Default pattern: `model-<NUM>-of-<NUM>.safetensors`
- Customizable through constructor parameters
- Efficient metadata caching via JSON

### Computation Pipeline
Our helper functions provide a streamlined approach to model operations:
- `compute_embedding`: Handles input embedding and causal mask setup
- `compute_layer`: Manages state transitions through decoder layers
- `compute_head`: Processes final linear projections and token prediction