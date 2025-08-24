from typing import Optional, Dict

import torch
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer

from llama_layer_collector.state_obj import LLmComputationState

mapper = {
    "llama": LlamaDecoderLayer,
    "qwen3": Qwen3DecoderLayer,
    "gemma3_text": Gemma3DecoderLayer
}

def getClass(config: PretrainedConfig) -> GradientCheckpointingLayer:
    return mapper[config.model_type]

class AutoDecoderLayer:
    def __init__(self, config: PretrainedConfig, layer_index: int):
        self.config = config
        if self.config._attn_implementation is None:
            self.config._attn_implementation = "eager"
        self.cls = getClass(self.config)(self.config, layer_index)

    def __call__(
        self, 
        state: LLmComputationState
    ) -> torch.Tensor:
        if self.cls.config.model_type == 'gemma3_text':
            return self.cls(
                hidden_states=state.state,
                attention_mask=state.causal_mask,
                position_ids=state.position_ids,
                past_key_values=None,
                cache_position=state.cache_position,
                position_embeddings_local=state.position_embeddings_local,
                position_embeddings_global=state.position_embeddings_global
            )[0]
        else:
            return self.cls(
                hidden_states=state.state,
                attention_mask=state.causal_mask,
                position_ids=state.position_ids,
                past_key_values=None,
                cache_position=state.cache_position,
                position_embeddings=state.position_embeddings
            )

    def to_empty(self, device: Optional[str]) -> 'AutoDecoderLayer':
        self.cls = self.cls.to_empty(device=device)
        return self

    def get_submodule(self, module_name: str):
        return self.cls.get_submodule(module_name)

    def to(self, device: str):
        self.cls = self.cls.to(device)
        return self