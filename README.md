Llama Layer Collector
=====================

![PyPI - Version](https://img.shields.io/pypi/v/llama-layer-collector)

**Llama Layer Collector** is a lightweight Python package for selectively loading and computing on individual layers of Llama-based language models. It is especially helpful when working with large, sharded checkpoints that you’d like to load only partially, or when you need granular access to model internals (embeddings, norms, decoder layers, etc.).

* * *

Key Features
------------

*   **Layer-by-Layer Loading:** Specify which layers to load (e.g., layers `0` through `10`) rather than loading the entire model.
*   **Caching for Speed:** Create and reuse cached metadata about shard files to avoid repeated scanning of checkpoints.
*   **Flexible Device & Precision Support:** Easily move layers to CPU or GPU and configure their precision (e.g., `torch.float16`).
*   **Helper Compute Functions:** Built-in utilities (e.g., `compute_embedding`, `compute_layer`, and `compute_head`) to perform partial or full forward passes without building an entire model class.

* * *

Installation
------------

You can install **Llama Layer Collector** directly from PyPI:

`pip install llama-layer-collector`

* * *

Class Overview: LlamaLayerCollector
-----------------------------------

The **LlamaLayerCollector** is initialized with several parameters that give you fine-grained control over how model layers are discovered and loaded:

*   **model\_dir (str)**  
    A required path to the directory containing model shards and a `config.json` file.
    
*   **cache\_file (str, optional)**  
    Path to a JSON file used for caching shard metadata. If no cache file is specified, the collector still builds metadata in memory but does not persist it for future runs.
    
*   **shard\_pattern (str, optional)**  
    A regular expression (default: `'model-(\\d+)-of-(\\d+).safetensors'`) indicating how shard files are named.
    
*   **layer\_prefix (str, optional)**  
    The string prefix identifying decoder layer keys in your model checkpoints (default: `'model.layers.'`).
    
*   **input\_embedding\_layer\_name (str, optional)**  
    Name of the input embedding weight parameter (default: `'model.embed_tokens.weight'`).
    
*   **norm\_layer\_name (str, optional)**  
    Name of the RMS norm layer weight parameter (default: `'model.norm.weight'`).
    
*   **lm\_head\_name (str, optional)**  
    Name of the LM head weight parameter (default: `'lm_head.weight'`).
    
*   **dtype (torch.dtype, optional)**  
    Data type (default: `torch.float16`) used when loading all model weights.
    
*   **device (str, optional)**  
    Device on which the loaded tensors will be placed (default: `'cpu'`, though `'cuda'` is common for GPU usage).
    

During initialization, the collector checks for a `config.json` file in `model_dir`. If the file is missing, a `FileNotFoundError` is raised.

### Commonly Used Methods

*   **`load_input_embedding()`**  
    Loads and returns a PyTorch `Embedding` layer for token embeddings.
    
*   **`load_norm()`**  
    Returns the RMSNorm layer (`LlamaRMSNorm` in Llama-based models) with loaded weights.
    
*   **`load_head()`**  
    Provides a linear layer for the LM head. If the head weights are not found, it defaults to using the input embedding weights.
    
*   **`load_layer_set(start_layer: int, end_layer: int)`**  
    Loads a specified range of decoder layers (e.g., from layer `0` to layer `5`), returning them as a list.
    

* * *

Example Usage
-------------

Below is a minimal example demonstrating how to load a Llama model’s layers individually, tokenize an input, and run a partial forward pass. This setup is particularly useful for memory-constrained environments or for debugging/tracing through specific model layers.

```python
from llama_layer_collector import LlamaLayerCollector
from llama_layer_collector.compute import compute_embedding, compute_layer, compute_head 
from transformers import AutoTokenizer  

# Specify the directory containing your model checkpoints and configuration. 
model_directory = "/path/to/llama/model"  
cache_file = "model_cache.json"    
# Create a collector instance with desired settings. 
collector = LlamaLayerCollector(     
    model_dir=model_directory,     
    cache_file=cache_file,     
    device="cuda",  # or "cpu"     
    dtype=torch.float16 
)  
# Load tokenizer from Transformers. 
tokenizer = AutoTokenizer.from_pretrained(model_directory) 
input_ids = tokenizer("The quick brown fox ", return_tensors='pt')['input_ids']  

# Load the input embedding layer. 
embedding = collector.load_input_embedding()  

# Load the normalization layer. 
norm = collector.load_norm()  

# Load the LM head (fallbacks to embedding if not available). 
head = collector.load_head()  

# Load a set of decoder layers (in this example, all layers). 
layers = collector.load_layer_set(0, collector.num_layers)  

# Perform a forward pass using the helper computation functions. 
state = compute_embedding(embedding, input_ids, collector.config) 
for lyr in layers:     
    state.state = compute_layer(lyr, state)

# Compute final output logits and retrieve the top predicted token ID. 
result = compute_head(head, norm(state.state), topk=1) 
print(f'Top predicted token ID: {result}')
```
1.  **Initialize the Collector**:  
    The `LlamaLayerCollector` scans your model directory, identifies shard files, and (optionally) caches metadata for fast reuse. 
2.  **Load Model Pieces**:  
    Grab individual components (embeddings, normalization, head, and a range of layers) as needed. 
3.  **Partial or Full Computation**:  
    Use the provided functions in `llama_layer_collector.compute` to sequentially pass data through each layer. This is especially handy for stepping through intermediate activations or customizing layer outputs.
4.  **Retrieve Predictions**:  
    Pass the final hidden state through the LM head, apply a softmax, and retrieve top-k token IDs.
    

* * *

When to Use This Package
------------------------

*   **Memory Constraints**: If your environment cannot hold an entire Llama model in memory, load only the layers you need.
*   **Debugging**: Trace the forward pass one layer at a time for analyzing intermediate states.
*   **Research & Development**: Experiment with custom modifications to specific layers or partial fine-tuning without instantiating the full model.

* * *

Additional Notes
----------------

*   **Shard Pattern**: By default, we look for files named `model-<NUM>-of-<NUM>.safetensors`. You can override this pattern in the constructor if your files follow a different naming convention.
*   **Caching**: A JSON cache file (e.g., `model_cache.json`) is automatically created and updated by the collector for quick retrieval of shard file information.
*   **Helper Compute Functions**:
    *   `compute_embedding`: Prepares the input embedding state and sets up the causal mask.
    *   `compute_layer`: Passes the current hidden state through a `LlamaDecoderLayer`.
    *   `compute_head`: Applies the final linear head to generate logits, then returns the top token(s).

* * *

Contributing
------------

Feedback, bug reports, and pull requests are welcome! Please open an issue or submit a PR on GitHub if you have any ideas for improvements or new features.

* * *

License
-------

This project is released under the MIT License.