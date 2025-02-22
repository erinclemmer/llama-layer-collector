import gc
from typing import List, Dict

from safetensors import safe_open
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig

def files_to_load_for_layer(
        layer_prefix: str,
        layer_file_cache: dict,
    ) -> List[str]:
    files_to_load = []
    for key in layer_file_cache.keys():
        if key.startswith(layer_prefix) and layer_file_cache[key] not in files_to_load:
            files_to_load.append(layer_file_cache[key])
    if len(files_to_load) == 0:
        raise Exception("Could not find layer data for layer prefix " + layer_prefix)
    return files_to_load

def files_to_load_for_layers(
        start_layer: int,
        end_layer: int,
        layer_prefix: str,
        layer_file_cache: dict
    ) -> List[str]:
    files_to_load = []
    for i in range(start_layer, end_layer+1):
        for f in files_to_load_for_layer(f'{layer_prefix}{i}.', layer_file_cache):
            if f not in files_to_load:
                files_to_load.append(f)
    return files_to_load

def load_layers(
        start_layer: int, 
        end_layer: int, 
        layer_prefix: str,
        layer_file_cache: Dict[str, str],
        config: LlamaConfig,
        model_dir: str,
        device: str,
        dtype: str
    ) -> List[LlamaDecoderLayer]:
    prefixes = [f'{layer_prefix}{i}.' for i in range(start_layer, end_layer+1)]
    shard_data = { }
    for file_path in files_to_load_for_layers(start_layer, end_layer, layer_prefix, layer_file_cache):
        full_path = f'{model_dir}/{file_path}'
        shard: dict = safe_open(full_path, framework='pt', device=device)
        for key in shard.keys():
            for prefix in prefixes:
                if key.startswith(prefix):
                    shard_data[key] = shard.get_tensor(key).detach().to(dtype)
        del shard
        gc.collect()
    
    layers = []
    for i in range(start_layer, end_layer+1):
        lyr = LlamaDecoderLayer(config, i).to(dtype=dtype)
        layer_data = { }
        for key in shard_data.keys():
            if key.startswith(f'{layer_prefix}{i}.'):
                layer_data[key.replace(f'{layer_prefix}{i}.', '')] = shard_data[key].detach()
        lyr.load_state_dict(layer_data)
        layers.append(lyr)

    return layers