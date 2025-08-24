from typing import Tuple

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

mapper = {
    "llama": LlamaRotaryEmbedding,
    "qwen3": Qwen3RotaryEmbedding
}

def getClass(config: PretrainedConfig) -> torch.nn.Module:
    return mapper[config.model_type]

class AutoRotaryEmbedding:
    def __init__(self, config: PretrainedConfig):
        self.config = config
        self.cls = getClass(config)(config)

    def __call__(self, x, position_ids) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cls(x, position_ids)
