from __future__ import annotations

import torch
from torch import Tensor, nn


class RationaleDecoder(nn.Module):
    """Two-layer decoder with a separate vocabulary embedding tied to its output projection."""

    def __init__(
        self,
        backbone_hidden_size: int,
        vocab_size: int,
        hidden_size: int = 128,
        attention_heads: int = 4,
        feed_forward_size: int = 512,
        dropout: float = 0.1,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.memory_projection = nn.Linear(backbone_hidden_size, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=feed_forward_size,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        memory: Tensor,
        *,
        target_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        target = self.token_embedding(input_ids)
        projected_memory = self.memory_projection(memory)
        return self.output_projection(
            self.decoder(
                tgt=target,
                memory=projected_memory,
                tgt_key_padding_mask=target_key_padding_mask,
            )
        )

    def greedy_decode(self, memory: Tensor, bos_token_id: int, eos_token_id: int, max_tokens: int) -> Tensor:
        generated = torch.full((memory.size(0), 1), bos_token_id, dtype=torch.long, device=memory.device)
        for _ in range(max_tokens - 1):
            logits = self.forward(generated, memory)[:, -1]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if bool(torch.all(next_token == eos_token_id)):
                break
        return generated
