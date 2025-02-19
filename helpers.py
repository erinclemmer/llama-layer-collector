import os
from typing import Dict

import torch
from safetensors import safe_open

def layers_to_load_by_memory(
        start_layer: int,
        max_memory: int,
        num_layers: int,
        layer_size_cache: Dict[str, int]
    ) -> int:
    current_layer = start_layer
    free_memory = max_memory
    next_layer_size = layer_size_cache[current_layer]
    while True:
        current_layer += 1
        if current_layer >= num_layers:
            break
        free_memory -= next_layer_size
        next_layer_size = layer_size_cache[current_layer]
        if free_memory - next_layer_size < 0:
            break
    end_layer = current_layer - 1
    return end_layer

def load_shard_tensor(
        layer_file_cache: dict, 
        model_dir: str,
        layer_name: str, 
        device: str,
        dtype: torch.dtype
    ) -> torch.Tensor:
    if layer_name not in layer_file_cache:
        raise ValueError(f'Could not find layer file for layer {layer_name}')
    file = layer_file_cache[layer_name]
    shard: dict = safe_open(os.path.join(model_dir, file), framework='pt', device=device)
    return shard.get_tensor(layer_name).to(dtype)