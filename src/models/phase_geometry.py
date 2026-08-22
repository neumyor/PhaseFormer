import torch
import torch.nn as nn


def build_circular_embedding(num_slots: int, dim: int, device=None, dtype=None):
    """Fixed Fourier embedding of the phase circle.

    Each slot p gets the fundamental-cycle angle ``angle = 2*pi*p/P`` and the
    slot embedding is built by repeating the orthogonal pair
    ``[sin(angle), cos(angle)]`` until `dim` features are filled. Unlike a
    learnable linear positional embedding, the circular geometry makes slot P
    wrap back to slot 0, encoding the periodic topology of the phase cycle
    (PhaseFormer experiment plan, stage 3).
    """
    p = torch.arange(num_slots, dtype=torch.float32, device=device)
    angle = 2.0 * torch.pi * p / num_slots  # (num_slots,)
    pairs = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)  # (num_slots, 2)
    emb = pairs.reshape(num_slots, -1)  # (num_slots, 2 * num_slots)
    if emb.shape[1] >= dim:
        emb = emb[:, :dim]
    else:
        reps = (dim + emb.shape[1] - 1) // emb.shape[1]
        emb = emb.repeat(1, reps)[:, :dim]
    if dtype is not None:
        emb = emb.to(dtype)
    return emb


class CircularPhaseEmbedding(nn.Module):
    """Holds the fixed circular phase embedding as a non-persistent buffer.

    Registered as a module so it follows ``.to(device)``. ``forward`` returns the
    embedding for the first `length` slots as ``(1, length, dim)`` so it can be
    broadcast onto a ``(B*C, L, D)`` latent in the same way as the learnable
    positional embedding.
    """

    def __init__(self, num_slots: int, dim: int):
        super().__init__()
        self.register_buffer(
            "embedding",
            build_circular_embedding(num_slots, dim),
            persistent=False,
        )

    def forward(self, length: int):
        return self.embedding[:length].unsqueeze(0)  # (1, length, dim)
