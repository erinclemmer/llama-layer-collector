import os
import gc
import json
from typing import List, Dict, Optional

import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer, LlamaConfig

from llama_layer_collector.load_layer import load_layers
from llama_layer_collector.cache import build_cache_data
from llama_layer_collector.helpers import load_shard_tensor

class LlamaLayerCollector:
    layer_prefix: str
    norm_layer_name: str
    input_embedding_layer_name: str
    lm_head_name: str
    shard_pattern: str
    
    config: LlamaConfig
    
    model_dir: str
    cache_file: str

    num_layers: int
    num_shards: int
    dtype: torch.dtype
    device: str
    layer_files: Dict[str, str]

    def __init__(
            self, 
            model_dir: str,
            cache_file: str = None,
            shard_pattern: str = r'model-(\d+)-of-(\d+).safetensors',
            layer_prefix: str = 'model.layers.',
            input_embedding_layer_name: str = 'model.embed_tokens.weight',
            norm_layer_name: str = 'model.norm.weight',
            lm_head_name: str = 'lm_head.weight',
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
        self.cache_file = cache_file
        
        self.lm_head_name = lm_head_name
        self.layer_prefix = layer_prefix
        self.norm_layer_name = norm_layer_name
        self.input_embedding_layer_name = input_embedding_layer_name
        self.shard_pattern = shard_pattern

        self.dtype = dtype
        self.device = device
        self.layer_files = { }
        if self.cache_file is None or not os.path.exists(self.cache_file):
            self._build_cache()
        else:
            self._read_cache()

    def _read_cache(self):
        if not os.path.exists(self.cache_file):
            raise FileNotFoundError('Could not find cache file ' + self.cache_file)
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            self.layer_files = json.load(f)

    def _build_cache(self):
        self.layer_files = build_cache_data(self.model_dir, self.shard_pattern, self.device)
        if self.cache_file is not None:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.layer_files, f, indent=4)

    def _load_shard_tensor(self, layer_name: str, device: str) -> torch.Tensor:
        return load_shard_tensor(self.layer_files, self.model_dir, layer_name, device, self.dtype)

    def load_input_embedding(self, device: str = None) -> torch.nn.Embedding:
        device = self.device if device is None else device
        return torch.nn.Embedding.from_pretrained(self._load_shard_tensor(self.input_embedding_layer_name, device))
    
    def load_norm(self, device: str = None) -> LlamaRMSNorm:
        device = self.device if device is None else device
        norm = LlamaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        norm.weight = torch.nn.Parameter(self._load_shard_tensor(self.norm_layer_name, device))
        return norm
    
    def load_head(self, device: str = None) -> torch.nn.Linear:
        device = self.device if device is None else device
        weight = None
        
        if self.lm_head_name is None or not self.lm_head_name in self.layer_files:
            weight = self.load_input_embedding(device).weight
        else:
            weight = self._load_shard_tensor(self.lm_head_name, device)

        head = torch.nn.Linear(weight.size()[1], weight.size()[0], device=device, dtype=self.dtype)
        head.weight = torch.nn.Parameter(weight)
        return head

    def load_layer_set(self, start_layer: int, end_layer: int, device: Optional[str] = None) -> List[LlamaDecoderLayer]:
        device = self.device if device is None else device
        layers = []
        for i in range(start_layer, end_layer+1, 3):
            layers.extend(load_layers(min(i, end_layer), min(i+2, end_layer), self.layer_prefix, self.layer_files, self.config, self.model_dir, device, self.dtype))
        gc.collect()
        return layers
