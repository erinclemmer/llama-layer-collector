import os
from pathlib import Path
import json
from typing import List, Dict

import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer, LlamaConfig

from load_layer import load_layers
from cache import build_cache_data
from helpers import layers_to_load_by_memory, load_shard_tensor

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

        
        self.model_dir = model_dir
        self.data_dir = data_dir
        
        if not os.path.exists(self.data_dir):
            Path(self.data_dir).mkdir(parents=True)
        
        self.lm_head_name = lm_head_name
        self.layer_prefix = layer_prefix
        self.norm_layer_name = norm_layer_name
        self.input_embedding_layer_name = input_embedding_layer_name
        self.verbose = verbose
        self.shard_pattern = shard_pattern

        self.dtype = dtype
        self.device = device
        self.layer_files = { }
        self.layer_size_cache = []
        if not os.path.exists(self._cache_file()):
            self._build_cache()
        self._read_cache()

    def _cache_file(self) -> str:
        return os.path.join(self.data_dir, CACHE_FILE_NAME)

    def _read_cache(self):
        cache_file = os.path.join(self.data_dir, CACHE_FILE_NAME)
        if not os.path.exists(cache_file):
            raise FileNotFoundError('Could not find cache file ' + cache_file)
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
    
        self.layer_files = cache_data['layer_files']
        self.layer_size_cache = cache_data['layer_sizes']

    def _build_cache(self):
        if self.verbose:
            print('Building cache')
        
        cache_data = build_cache_data(self.model_dir, self.shard_pattern, self.dtype, self.device, self.config, self.verbose)
        
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        with open(self._cache_file(), 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=4)

    def _load_shard_tensor(self, layer_name: str, device: str) -> torch.Tensor:
        return load_shard_tensor(self.layer_files, self.model_dir, layer_name, device, self.dtype)

    def _layers_to_load_by_memory(self, start_layer: int, max_memory: int) -> int:
        return layers_to_load_by_memory(start_layer, max_memory, self.num_layers, self.layer_size_cache)

    def input_embedding(self, device: str = None) -> torch.nn.Embedding:
        if self.verbose:
            print('Loading input embedding')
        
        device = self.device if device is None else device
        return torch.nn.Embedding.from_pretrained(self._load_shard_tensor(self.input_embedding_layer_name, device))
    
    def norm(self, device: str = None) -> LlamaRMSNorm:
        if self.verbose:
            print('Loading norm')
        device = self.device if device is None else device
        norm = LlamaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        norm.weight = torch.nn.Parameter(self._load_shard_tensor(self.norm_layer_name, device))
        return norm
    
    def head(self, device: str = None):
        if self.verbose:
            print('Loading head')
        device = self.device if device is None else device
        weight = None
        
        if self.lm_head_name is None or not self.lm_head_name in self.layer_files:
            weight = self.input_embedding(device).weight
        else:
            weight = self._load_shard_tensor(self.lm_head_name, device)

        head = torch.nn.Linear(weight.size()[1], weight.size()[0], device=device, dtype=self.dtype)
        head.weight = torch.nn.Parameter(weight)
        return head

    def load_layer_set(self, start_layer: int, end_layer: int, device: str = None) -> List[LlamaDecoderLayer]:
        device = self.device if device is None else device
        return load_layers(start_layer, end_layer, self.layer_prefix, self.layer_files, self.config, self.data_dir, device, self.dtype)

    def load_layers_by_memory(self, start_layer: int, max_memory: int, device: str = None) -> List[LlamaDecoderLayer]:
        end_layer = layers_to_load_by_memory(start_layer, max_memory, self.num_layers, self.layer_size_cache)
        return self.load_layer_set(start_layer, end_layer, device)