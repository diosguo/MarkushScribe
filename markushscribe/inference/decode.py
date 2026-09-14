"""Greedy autoregressive token decoding for the NODE AR model."""

from __future__ import annotations

import torch

from ..tokenizer import DecodedNode, Vocab, decode_tokens


@torch.no_grad()
def greedy_decode_tokens(
    model,
    image: torch.Tensor,
    vocab: Vocab,
    *,
    max_len: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Greedily decode one image into a ``[1, L]`` token stream."""
    model.eval()
    device = device or image.device
    encoder_out = model.encoder(image.to(device))
    limit = max_len or model.config.max_len
    tokens = torch.tensor([[vocab.bos_id]], device=device, dtype=torch.long)
    for _ in range(limit):
        hidden, logits = model.node_ar(tokens, encoder_out.memory)
        next_token = int(logits[:, -1].argmax(dim=-1).item())
        tokens = torch.cat(
            [tokens, torch.tensor([[next_token]], device=device, dtype=torch.long)], dim=1
        )
        if next_token == vocab.eos_id:
            break
    return tokens


def decode_nodes(token_ids: list[int] | torch.Tensor, vocab: Vocab) -> list[DecodedNode]:
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.flatten().tolist()
    return decode_tokens(list(token_ids), vocab)


__all__ = ["decode_nodes", "greedy_decode_tokens"]
