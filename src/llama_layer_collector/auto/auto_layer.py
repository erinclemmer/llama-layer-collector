from typing import Optional, Dict

import torch
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

mapper = {
    "llama": LlamaDecoderLayer,
    "qwen3": Qwen3DecoderLayer
}

def getClass(config: PretrainedConfig) -> GradientCheckpointingLayer:
    return mapper[config.model_type]

class AutoDecoderLayer:
    def __init__(self, config: PretrainedConfig, layer_index: int):
        self.config = config
        self.cls = getClass(config)(config, layer_index)

    def __call__(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> torch.Tensor:
        return self.cls(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings
        )

    def to_empty(self, device: Optional[str]) -> 'AutoDecoderLayer':
        self.cls = self.cls.to_empty(device=device)
        return self

    def get_submodule(self, module_name: str):
        return self.cls.get_submodule(module_name)

    def to(self, device: str):
        self.cls = self.cls.to(device)
        return self