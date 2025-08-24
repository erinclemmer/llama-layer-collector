import copy
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask

from llama_layer_collector.auto.auto_rotary import AutoRotaryEmbedding
from llama_layer_collector.auto.auto_layer import AutoDecoderLayer
from llama_layer_collector.state_obj import LLmComputationState

# TODO Allow use_cache
def compute_embedding(
        input_embedder: torch.nn.Embedding,
        input_ids: torch.Tensor,
        config: PretrainedConfig
    ) -> LLmComputationState:
    device = input_embedder.weight.device
    embedded_input = input_embedder(input_ids.to(device))
    state = LLmComputationState()
    state.state = embedded_input
    state.cache_position = torch.arange(0, end=embedded_input.size(1), device=device)
    state.position_ids = state.cache_position.unsqueeze(0)
    state.causal_mask = create_causal_mask(
        config=config,
        input_embeds=embedded_input.detach(),
        attention_mask=None,
        cache_position=state.cache_position,
        past_key_values=None,
        position_ids=state.position_ids
    )
    if config.model_type == 'gemma3_text':
        state.position_embeddings_global = AutoRotaryEmbedding(config)(embedded_input.detach(), state.position_ids)
        configCopy = copy.deepcopy(config)
        configCopy.rope_theta = configCopy.rope_local_base_freq
        configCopy.rope_scaling = {"rope_type": "default"}
        
        state.position_embeddings_local = AutoRotaryEmbedding(configCopy)(embedded_input.detach(), state.position_ids)
    else:
        state.position_embeddings = AutoRotaryEmbedding(config)(embedded_input.detach(), state.position_ids)
    return state

def compute_head(
        head: torch.nn.Linear,
        state: torch.Tensor,
        topk: int = 1
    ) -> torch.Tensor:
    state = head(state[:, -1, :])
    probs = torch.softmax(state, dim=-1)
    return torch.topk(probs, topk).indices
