import torch

class LLmComputationState:
    state: torch.Tensor
    position_embeddings: torch.Tensor
    position_embeddings_local: torch.Tensor
    position_embeddings_global: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    causal_mask: torch.Tensor