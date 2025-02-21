import re
import os
import gc
from typing import List

import torch
from safetensors import safe_open
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer

def size_of_tensor(t: torch.Tensor):
    return t.element_size() * t.nelement()

def get_size_of_layer(layer_idx: int, dtype: torch.dtype, config: LlamaConfig) -> int:
    if layer_idx < 0 or layer_idx > config.num_hidden_layers:
            raise ValueError('Layer index out of bounds')
    lyr = LlamaDecoderLayer(config, layer_idx).to(dtype=dtype)
    tensors = [
        lyr.self_attn.q_proj.weight,
        lyr.self_attn.k_proj.weight,
        lyr.self_attn.v_proj.weight,
        lyr.self_attn.o_proj.weight,
        lyr.mlp.gate_proj.weight,
        lyr.mlp.up_proj.weight,
        lyr.mlp.down_proj.weight,
        lyr.post_attention_layernorm.weight
    ]
    return sum([size_of_tensor(t) for t in tensors])

def get_shard_files(shard_pattern: str, model_dir: str) -> List[str]:
    if 'model.safetensors' in os.listdir(model_dir):
        return ['model.safetensors']
    
    multiple_pattern = re.compile(shard_pattern)
    shard_files = [f for f in os.listdir(model_dir) if multiple_pattern.match(f)]
    if not shard_files:
        raise Exception("No Shard files in specified directory " + model_dir)

    shard_files.sort()
    return shard_files

def build_cache_data(
        model_dir: str,
        shard_pattern: str,
        dtype: torch.dtype,
        device: str,
        config: LlamaConfig
    ):
    layer_files = { }
    for file in get_shard_files(shard_pattern, model_dir):
        full_path = os.path.join(model_dir, file)
        shard: dict = safe_open(full_path, framework='pt', device=device)
        for key in shard.keys():
            layer_files[key] = file
        del shard
        gc.collect() # is this necessary?

    cache_data = {
        "layer_files": layer_files,
        "layer_sizes": [get_size_of_layer(i, dtype, config) for i in range(0, config.num_hidden_layers)]
    }
    return cache_data