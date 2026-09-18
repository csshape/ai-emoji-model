"""A deliberately small transformer encoder for multi-label emoji prediction.

Written with plain ops rather than ``nn.TransformerEncoder`` so the graph
exports cleanly to ONNX/Core ML -- the built-in layer has a nested-tensor fast
path that frequently breaks tracing.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    n_emoji: int = 512
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    max_len: int = 64
    dropout: float = 0.1
    label_head: bool = False      # tie the output layer to emoji descriptions
    label_len: int = 24           # tokens kept per emoji description

    def to_dict(self) -> dict:
        return asdict(self)


class EncoderLayer(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, t, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        att = att.transpose(1, 2).reshape(b, t, d)
        x = x + self.dropout(self.proj(att))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class EmojiEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        if cfg.label_head:
            # Output weights are derived from each emoji's description instead
            # of being free parameters, so text and labels share one space.
            self.register_buffer("label_tokens",
                                 torch.zeros((cfg.n_emoji, cfg.label_len), dtype=torch.long))
            self.label_bias = nn.Parameter(torch.zeros(cfg.n_emoji))
            self.label_scale = nn.Parameter(torch.tensor(4.0))
            self.label_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.head = None
        else:
            self.head = nn.Linear(cfg.d_model, cfg.n_emoji)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def set_label_tokens(self, tokens: torch.Tensor) -> None:
        """Install tokenised emoji descriptions, shape (n_emoji, label_len)."""
        self.label_tokens.copy_(tokens.to(self.label_tokens.dtype))

    def label_weights(self) -> torch.Tensor:
        """Mean-pool each description's token embeddings into an output vector."""
        mask = self.label_tokens.ne(0).unsqueeze(-1).float()
        emb = self.token_emb(self.label_tokens) * mask
        pooled = emb.sum(1) / mask.sum(1).clamp(min=1.0)
        return self.label_proj(pooled)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, T) int64, 0 = padding. Returns logits (B, n_emoji)."""
        pad = input_ids.eq(0)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

        # Additive attention bias: -inf on padded keys, broadcast over heads.
        bias = torch.zeros_like(pad, dtype=x.dtype).masked_fill(pad, float("-inf"))
        bias = bias[:, None, None, :]
        for layer in self.layers:
            x = layer(x, bias)
        x = self.norm(x)

        # Masked mean pool.
        keep = (~pad).to(x.dtype).unsqueeze(-1)
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        if self.head is not None:
            return self.head(pooled)
        w = F.normalize(self.label_weights(), dim=-1)
        return self.label_scale * F.normalize(pooled, dim=-1) @ w.t() + self.label_bias

    def bake_label_head(self) -> None:
        """Collapse the label head into a plain Linear for clean export."""
        if self.head is not None:
            return
        with torch.no_grad():
            w = F.normalize(self.label_weights(), dim=-1) * self.label_scale
            head = nn.Linear(self.cfg.d_model, self.cfg.n_emoji)
            head.weight.copy_(w)
            head.bias.copy_(self.label_bias)
        self.head = head

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
