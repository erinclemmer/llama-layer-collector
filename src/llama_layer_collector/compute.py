import torch
from llama_layer_collector.helpers import update_causal_mask
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig, LlamaRotaryEmbedding, LlamaRMSNorm

class LLmComputationState:
    state: torch.Tensor
    position_embeddings: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    causal_mask: torch.Tensor

def compute_embedding(
        input_embedder: torch.nn.Embedding,
        input_ids: torch.Tensor,
        config: LlamaConfig
    ) -> LLmComputationState:
    device = input_embedder.weight.device
    embedded_input = input_embedder(input_ids.to(device))
    state = LLmComputationState()
    state.state = embedded_input
    state.cache_position = torch.arange(0, end=embedded_input.size(1), device=device)
    state.causal_mask = update_causal_mask(config, embedded_input.detach().to(device), state.cache_position)
    state.position_ids = state.cache_position.unsqueeze(0)
    state.position_embeddings = LlamaRotaryEmbedding(config=config)(embedded_input.detach(), state.position_ids)
    return state

def compute_layer(
        lyr: LlamaDecoderLayer,
        state: LLmComputationState 
    ) -> torch.Tensor:
    return lyr(
        state.state,
        attention_mask=state.causal_mask,
        position_ids=state.position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=state.cache_position,
        position_embeddings=state.position_embeddings
    )[0]

def compute_head(
        head: torch.nn.Linear,
        state: torch.Tensor,
        topk: int = 1
    ) -> torch.Tensor:
    state = head(state[:, -1, :])
    probs = torch.softmax(state, dim=-1)
    return torch.topk(probs, topk).indices
