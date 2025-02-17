import re
import os
import gc
import json
from typing import List, Tuple, Dict

import torch
from safetensors import safe_open
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer, LlamaConfig

from util import size_of_tensor

CACHE_FILE_NAME = 'cache.json'

class LayerDataCollector:
    layer_prefix: str
    norm_layer_name: str
    input_embedding_layer_name: str
    lm_head_name: str
    shard_pattern: str
    
    config: LlamaConfig
    
    model_dir: str
    data_dir: str

    num_layers: int
    num_shards: int
    verbose: bool
    dtype: torch.dtype
    device: str
    layer_files: Dict[str, str]
    layer_sizes: Dict[str, int]

    def __init__(
            self, 
            model_dir: str,
            data_dir: str,
            shard_pattern: str = r'model-(\d+)-of-(\d+).safetensors',
            layer_prefix: str = 'model.layers.',
            input_embedding_layer_name: str = 'model.embed_tokens.weight',
            norm_layer_name: str = 'model.norm.weight',
            lm_head_name: str = 'lm_head.weight',
            verbose: bool = False,
            dtype: torch.dtype = torch.float16,
            device: str = 'cpu'
        ):
        config_file_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_file_path):
            raise FileNotFoundError('Could not find config file ' + config_file_path)
        
        with open(config_file_path, 'r', encoding='utf-8') as f:
            self.config = LlamaConfig.from_dict(json.load(f))
            self.num_layers = self.config.num_hidden_layers

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        
        self.model_dir = model_dir
        self.data_dir = data_dir
        
        self.lm_head_name = lm_head_name
        self.layer_prefix = layer_prefix
        self.norm_layer_name = norm_layer_name
        self.input_embedding_layer_name = input_embedding_layer_name
        self.verbose = verbose
        self.shard_pattern = shard_pattern

        self.dtype = dtype
        self.device = device
        self.layer_files = { }
        self.layer_size_cache = None
        self.build_cache()
        
    def get_shard_files(self, shard_pattern: str):
        if 'model.safetensors' in os.listdir(self.data_dir):
            return ['model.safetensors']
        
        multiple_pattern = re.compile(shard_pattern)
        shard_files = [f for f in os.listdir(self.data_dir) if multiple_pattern.match(f)]
        if not shard_files:
            raise Exception("No Shard files in specified directory " + self.data_dir)

        shard_files.sort()
        return shard_files

    def get_size_of_layer(self, layer_idx: int):
        if self.layer_size_cache is not None:
            return self.layer_size_cache[layer_idx]
        if self.verbose:
            print(f'Getting size of layer {layer_idx} of {self.num_layers}')
        lyr = LlamaDecoderLayer(self.config, layer_idx).to(dtype=self.dtype)
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

    def build_cache(self):
        cache_file = os.path.join(self.data_dir, CACHE_FILE_NAME)
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        
            self.layer_files = cache_data['layer_files']
            self.layer_size_cache = cache_data['layer_sizes']
            return
        
        if self.verbose:
            print('Building cache')
        layer_files = { }
        for file in self.get_shard_files(self.shard_pattern):
            if self.verbose:
                print('Loading ' + file)
            full_path = os.path.join(self.model_dir, file)
            shard: dict = safe_open(full_path, framework='pt', device=self.device)
            for key in shard.keys():
                layer_files[key] = file
            del shard
            gc.collect() # is this necessary?

        cache_data = {
            "layer_files": layer_files,
            "layer_sizes": [self.get_size_of_layer(i) for i in range(0, self.num_layers)]
        }

        self.layer_files = cache_data['layer_files']
        self.layer_sizes = cache_data['layer_sizes']
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=4)

    def _get_shard_layer(self, layer_name: str, device: str) -> torch.Tensor:
        file = self.layer_files[layer_name]
        shard: dict = safe_open(os.path.join(self.model_dir, file), framework='pt', device=device)
        return shard.get_tensor(layer_name).to(self.dtype)

    def load_input_embedding(self, device: str = None) -> torch.nn.Embedding:
        if self.verbose:
            print('Loading input embedding')
        
        device = self.device if device is None else device
        return torch.nn.Embedding.from_pretrained(self._get_shard_layer(self.input_embedding_layer_name, device))
    
    def load_norm(self, device: str = None) -> LlamaRMSNorm:
        if self.verbose:
            print('Loading norm')
        device = self.device if device is None else device
        norm = LlamaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        norm.weight = torch.nn.Parameter(self._get_shard_layer(self.norm_layer_name, device))
        return norm
    
    def load_head(self, device: str = None):
        if self.verbose:
            print('Loading head')
        device = self.device if device is None else device
        weight = None
        
        if self.lm_head_name is None or not self.lm_head_name in self.layer_files:
            weight = self.load_input_embedding(device).weight
        else:
            weight = self._get_shard_layer(self.lm_head_name, device)

        head = torch.nn.Linear(weight.size()[1], weight.size()[0], device=device, dtype=self.dtype)
        head.weight = torch.nn.Parameter(weight)
        return head

    def layer_prefix(self, layer_idx: int) -> str:
        return f'{self.layer_prefix}{layer_idx}.'

    def _get_files_to_load_for_layer(self, layer_idx: int) -> List[str]:
        layer_prefix = self.layer_prefix(layer_idx)
        files_to_load = []
        for key in self.layer_files.keys():
            if key.startswith(layer_prefix) and self.layer_files[key] not in files_to_load:
                files_to_load.append(self.layer_files[key])
        if len(files_to_load) == 0:
            raise Exception("Could not find layer data for layer " + str(layer_idx))
        return files_to_load
    
    def _get_files_start_end(self, start_layer: int, end_layer: int):
        files_to_load = []
        for i in range(start_layer, end_layer+1):
            for f in self._get_files_to_load_for_layer(i):
                if f not in files_to_load:
                    files_to_load.append(f)
        return files_to_load
    
    def _get_end_layer_for_memory(self, start_layer: int, max_memory: int) -> int:
        current_layer = start_layer
        free_memory = max_memory
        next_layer_size = self.get_size_of_layer(current_layer)
        while True:
            current_layer += 1
            if current_layer >= self.num_layers:
                break
            free_memory -= next_layer_size
            next_layer_size = self.get_size_of_layer(current_layer)
            if free_memory - next_layer_size < 0:
                break
        end_layer = current_layer - 1
        return end_layer

    def _get_layer_data(self, files_to_load: List[str], start_layer, end_layer: int, device: str) -> dict:
        device = self.device if device is None else device
        prefixes = [f'{self.layer_prefix}{i}' for i in range(start_layer, end_layer+1)]
        layer_data = { }
        for file_path in files_to_load:
            full_path = os.path.join(self.data_dir, file_path)
            shard: dict = safe_open(full_path, framework='pt', device=device)
            for key in shard.keys():
                for prefix in prefixes:
                    if key.startswith(prefix):
                        layer_data[key] = shard.get_tensor(key).detach().to(self.dtype)
            del shard
            gc.collect()
        return layer_data

    def _get_start_end_layers_from_dict(self, layer_data: dict) -> Tuple[int, int]:
        min_layer = int('inf')
        max_layer = -1
        for key in layer_data.keys():
            if not self.layer_prefix in key:
                continue
            try:
                match = re.search(f'{self.layer_prefix}(\\d+)', key)
                if match:
                    s = match.group(1)
                    layer_number = int(s)
                    if layer_number > max_layer:
                        max_layer = layer_number
                    if layer_number < min_layer:
                        min_layer = layer_number
            except:
                continue
        return min_layer, max_layer

    def _dict_to_layers(self, layer_data: dict) -> List[LlamaDecoderLayer]:
        layers = []
        start_layer, end_layer = self._get_start_end_layers_from_dict(layer_data)
        for i in range(start_layer, end_layer+1):
            layer_prefix = self.layer_prefix(i)
            lyr = LlamaDecoderLayer(self.config, i).to(dtype=self.dtype)
            layer_data = { }
            for key in layer_data.keys():
                if key.startswith(layer_prefix):
                    layer_data[key.replace(layer_prefix, '')] = layer_data[key].detach()
            lyr.load_state_dict(layer_data)
            layers.append(lyr)
        return layers

    def load_layer_set(self, start_layer: int, end_layer: int, device: str = None) -> List[LlamaDecoderLayer]:
        files_to_load = self._get_files_start_end(start_layer, end_layer)
        
        data = self._get_layer_data(files_to_load, start_layer, end_layer, device)
        return self._dict_to_layers(data)

    def load_layers_by_memory(self, start_layer: int, max_memory: int, device: str = None) -> List[LlamaDecoderLayer]:
        if end_layer is None:
            end_layer = self._get_end_layer_for_memory(start_layer, max_memory)
        
        return self.load_layer_set(start_layer, end_layer, device)