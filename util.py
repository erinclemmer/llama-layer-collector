import io

import torch
from hashlib import sha256

def size_of_tensor(t: torch.Tensor):
    return t.element_size() * t.nelement()

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    bts = io.BytesIO()
    torch.save(t, bts)
    return bts.getvalue()

def get_hash(data: bytes) -> str:
    return sha256(data).hexdigest()

def tensor_hash(t: torch.Tensor) -> str:
    return get_hash(tensor_to_bytes(t))
