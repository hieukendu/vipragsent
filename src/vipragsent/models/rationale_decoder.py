from __future__ import annotations

import torch
from torch import Tensor, nn


class RationaleDecoder(nn.Module):
    """Causal two-layer decoder with a separately tied decoder vocabulary."""

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
        memory_key_padding_mask: Tensor | None = None,
        causal_mask: Tensor | None = None,
    ) -> Tensor:
        target = self.token_embedding(input_ids)
        projected_memory = self.memory_projection(memory)
        if causal_mask is None:
            causal_mask = torch.triu(torch.ones((input_ids.size(1), input_ids.size(1)), dtype=torch.bool, device=input_ids.device), diagonal=1)
        return self.output_projection(self.decoder(
            tgt=target,
            memory=projected_memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        ))

    def teacher_forcing(
        self,
        target_ids: Tensor,
        target_attention_mask: Tensor,
        memory: Tensor,
        memory_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if target_ids.ndim != 2 or target_attention_mask.shape != target_ids.shape:
            raise ValueError("Target IDs and target attention mask must have the same [batch, time] shape")
        if target_ids.size(1) < 2:
            raise ValueError("Teacher forcing requires BOS and at least one target token")
        decoder_input = target_ids[:, :-1]
        decoder_input_attention = target_attention_mask[:, :-1].bool()
        labels = target_ids[:, 1:].clone()
        labels = labels.masked_fill(~target_attention_mask[:, 1:].bool(), -100)
        target_padding = ~decoder_input_attention
        memory_padding = ~memory_attention_mask.bool()
        causal = torch.triu(torch.ones((decoder_input.size(1), decoder_input.size(1)), dtype=torch.bool, device=target_ids.device), diagonal=1)
        logits = self.forward(
            decoder_input,
            memory,
            target_key_padding_mask=target_padding,
            memory_key_padding_mask=memory_padding,
            causal_mask=causal,
        )
        return logits, labels, target_padding

    @torch.no_grad()
    def greedy_decode(self, memory: Tensor, memory_attention_mask: Tensor, bos_token_id: int, eos_token_id: int, max_tokens: int) -> Tensor:
        if max_tokens < 2:
            raise ValueError("max_tokens must allow BOS and EOS")
        generated = torch.full((memory.size(0), 1), bos_token_id, dtype=torch.long, device=memory.device)
        finished = torch.zeros(memory.size(0), dtype=torch.bool, device=memory.device)
        for _ in range(max_tokens - 1):
            logits = self.forward(generated, memory, memory_key_padding_mask=~memory_attention_mask.bool())[:, -1]
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            finished |= next_token == eos_token_id
            if bool(torch.all(finished)):
                break
        return generated
