Llama Model Layer Loader
========================

A lightweight Python utility to selectively load layers from sharded Llama model checkpoints. This project facilitates efficient loading and caching of model components, enabling flexible layer-level manipulation and inference for Llama-based models.

* * *

Table of Contents
-----------------

*   [Introduction](#introduction)
*   [Features](#features)
*   [Installation](#installation)
*   [Usage](#usage)
*   [Internal Layer Loading Mechanism](#internal-layer-loading-mechanism)
*   [Helper Computation Functions](#helper-computation-functions)
*   [Testing](#testing)
*   [Code Structure](#code-structure)
*   [Customization](#customization)
*   [Contributing](#contributing)
*   [License](#license)

* * *

Introduction
------------

The Llama Model Layer Loader is designed to load specific layers from Llama model checkpoints that are split across multiple shards. It uses a caching mechanism to expedite subsequent loads and provides helper methods to extract key model components including:

*   **Input Embedding Layer**
*   **Normalization Layer**
*   **LM Head**
*   **Decoder Layers**

This project is particularly useful in scenarios where memory constraints or custom processing require loading only parts of the full model.

* * *

Features
--------

*   **Selective Layer Loading:**  
    Load a specific set of decoder layers by specifying start and end indices.
    
*   **Caching Mechanism:**  
    Cache metadata (layer file paths and sizes) to speed up repeated loads.
    
*   **Flexible Device & Data Type Support:**  
    Specify the target device (CPU, GPU) and precision (e.g., `torch.float16`).
    
*   **Multi-Model Support:**  
    The test suite demonstrates usage with both 1B and 8B model variants, ensuring the loader adapts to different model configurations.
    
*   **Robust Exception Handling:**  
    Comprehensive tests verify that the proper exceptions are raised for invalid inputs and missing files.
    
*   **Full Stack Operations:**  
    The loader has been validated in end-to-end scenarios where embedding layers, decoder layers, and head components are stacked to simulate text generation.
    
*   **Additional Helper Functions:**  
    Although not the core focus, helper functions in `compute.py` provide an easy way to compute on the loaded data, simplifying end-to-end testing and integration.
    

* * *

Installation
------------

### Prerequisites

*   Python 3.10 (this is the highest version that works with cuda support for pytorch)
*   PyTorch
*   [Transformers](https://github.com/huggingface/transformers)
*   [Safetensors](https://github.com/huggingface/safetensors)

### Setup

1.  **Clone the Repository:**
    
    bash
    
    Copy code
    
    `git clone https://github.com/your-username/llama-layer-loader.git cd llama-layer-loader`
    
2.  **Install Dependencies:**
    
    It is recommended to use a virtual environment. Then, install the required packages:
    
    bash
    
    Copy code
    
    `pip install torch transformers safetensors`
    
3.  **Project Files:**
    
    Ensure the following modules are present:
    
    *   `llama_layer_collector.py` (contains the `LlamaLayerCollector` class)
    *   `load_layer.py` (functions for identifying and loading specific model layers)
    *   `cache.py` (for building cache data)
    *   `helpers.py` (for loading shard tensors and other utility functions)
    *   `compute.py` (helper functions for performing computations on the loaded data)
    *   A valid `config.json` file in the model directory.

* * *

Usage
-----

Below is a simple example demonstrating how to use the `LlamaLayerCollector`:
```python
from llama_layer_collector import LlamaLayerCollector  # Specify the directory containing your model checkpoints and configuration. 

model_directory = "/path/to/llama/model" 
cache_file = "model_cache.json"  
# Create a collector instance with desired settings. 
collector = LlamaLayerCollector(     
    model_dir=model_directory,     
    cache_file=cache_file,     
    device="cuda",  # or "cpu"     
    dtype=torch.float16
)  
# Load the input embedding layer. 
embedding = collector.load_input_embedding()  

# Load the normalization layer. 
norm = collector.load_norm()  

# Load the LM head (fallback to input embedding weights if not available). 
head = collector.load_head()  

# Load a set of decoder layers, for example, all layers. 
layers = collector.load_layer_set(0, collector.num_layers)  

# Now you can integrate these components into your model architecture.
```

* * *

Internal Layer Loading Mechanism
--------------------------------

The core functionality for loading model layers is implemented in `load_layer.py`. Key components include:

*   **File Identification:**
    
    *   `files_to_load_for_layer`: Scans a cache dictionary for shard files containing data for a given layer prefix.
    *   `files_to_load_for_layers`: Aggregates shard files for a specified range of layers.
*   **Layer Loading:**
    
    *   `load_layers`:
        *   Constructs prefixes for each desired decoder layer.
        *   Opens the corresponding shard files using the `safetensors` library.
        *   Extracts tensor data for each layer and constructs a state dictionary.
        *   Initializes instances of `LlamaDecoderLayer` and loads weights from the state dictionary.
*   **Memory Management:**
    
    *   Explicit deletion and garbage collection are used after processing each shard to optimize memory usage.

This modular approach ensures that only the required layers are loaded into memory, making the process both efficient and scalable.

* * *

Helper Computation Functions
----------------------------

While the primary focus of this project is on loading model layers, the repository also includes helper functions in `compute.py` that allow for easy computation on the loaded data. These functions are designed to simplify testing and end-to-end usage:

*   **`compute_embedding`:**  
    Computes the initial embedding state from input token IDs using the provided input embedding module. This function also sets up the causal mask and rotary embeddings necessary for subsequent processing.
    
*   **`compute_layer`:**  
    Applies a specified decoder layer to the current computation state, facilitating layer-by-layer forward passes.
    
*   **`compute_head`:**  
    Computes the output logits using the LM head and returns the top-k predictions via softmax.
    

These helpers integrate with other parts of the project (such as functions in `helpers.py`) and serve as an easy-to-use interface for performing computations on the loaded layers.

* * *

Testing
-------

The project includes a comprehensive test suite (`test.py`) that demonstrates:

*   **Multi-Model Support:**  
    Tests cover both 1B and 8B Llama model variants, ensuring that cache files are generated correctly and that the expected number of keys and layer sizes are present.
    
*   **Component Validation:**  
    Tests verify that functions such as `load_input_embedding`, `load_norm`, `load_head`, and `load_layer_set` return tensors and modules with the expected shapes.
    
*   **End-to-End Processing:**  
    Simulated text generation tests show that the entire model stack—embedding, decoder layers, and head—can be used sequentially to generate tokens, mimicking real-world inference scenarios.
    
*   **Robust Exception Handling:**  
    Various tests ensure that invalid parameters or missing files correctly trigger exceptions, ensuring the reliability of the loader.
    

### Running the Tests

To run the tests directly from the command line, execute:

bash

Copy code

`python test.py`

This will run all unit tests and provide output on the test results.

* * *

Code Structure
--------------

graphql

Copy code

`├── llama_layer_collector.py   # Contains the LlamaLayerCollector class. ├── load_layer.py              # Functions to identify shard files and load specific model layers. ├── cache.py                   # Utilities for building and managing cache data. ├── helpers.py                 # Helper functions, including shard tensor loading and utilities. ├── compute.py                 # Additional helper functions for performing computations on loaded data. ├── test.py                    # Comprehensive test suite for validating loader functionality. └── config.json                # Model configuration file (provided with your model checkpoint).`

Each component is modular, making it straightforward to update or replace parts of the system based on your specific requirements.

* * *

Customization
-------------

*   **Shard Pattern:**  
    The default regex pattern `r'model-(\d+)-of-(\d+).safetensors'` can be customized to match your shard file naming conventions.
    
*   **Layer Names:**  
    Modify the parameters `layer_prefix`, `input_embedding_layer_name`, `norm_layer_name`, and `lm_head_name` in the `LlamaLayerCollector` constructor to suit different model architectures or naming schemes.
    
*   **Device & Data Type:**  
    Change the target `device` (e.g., `"cpu"` or `"cuda"`) and the precision (default `torch.float16`) to match your hardware setup.
    

* * *

Contributing
------------

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request on the project's GitHub repository.

* * *

License
-------

This project is released under the MIT License.

* * *

Happy modeling! Enjoy efficient and flexible layer loading with Llama Model Layer Loader, and leverage the provided helper functions to easily compute on the loaded data.