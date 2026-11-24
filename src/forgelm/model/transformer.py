"""Complete ForgeLM Transformer causal language model with KV cache support."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from forgelm.model.block import TransformerBlock
from forgelm.model.config import ModelConfig

if TYPE_CHECKING:
    from forgelm.inference.kv_cache import KVCache


class ForgeTransformer(nn.Module):
    """Decoder-only Transformer causal language model trained from scratch.

    Features:
        - Token and learned positional embeddings
        - Stack of pre-LayerNorm Transformer blocks
        - Final LayerNorm and linear language model head
        - Weight tying between token embeddings and LM head
        - Special weight initialization for stable training
        - Key-Value (KV) Cache support for accelerated generation
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.d_model),
                "wpe": nn.Embedding(config.max_seq_len, config.d_model),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
                "ln_f": nn.LayerNorm(config.d_model),
            }
        )

        # Output projection head: projects d_model -> vocab_size
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2017): shares representation between input & output
        self.transformer["wte"].weight = self.lm_head.weight

        # Initialize all model weights
        self.apply(self._init_weights)

        # Apply special scaled initialization to residual output projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with truncated normal distribution for linear/embedding."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return the number of parameters in the model.

        Args:
            non_embedding: If True, excludes positional and token embedding parameters.

        Returns:
            Integer parameter count.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer["wpe"].weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for causal language model with optional KV cache.

        Args:
            idx: Tensor of token IDs of shape (batch_size, seq_len).
            targets: Optional tensor of target token IDs of shape (batch_size, seq_len).
            kv_cache: Optional KVCache instance storing past keys/values.
            start_pos: Starting position index for learned positional embeddings.

        Returns:
            Tuple of (logits, loss). Loss is None if targets is not provided.
            logits shape: (batch_size, seq_len, vocab_size).
        """
        device = idx.device
        b, t = idx.size()

        if start_pos + t > self.config.max_seq_len:
            raise ValueError(
                f"Position {start_pos + t} exceeds maximum context length {self.config.max_seq_len}"
            )

        # Position indices: [start_pos, start_pos + 1, ..., start_pos + t - 1]
        pos = torch.arange(start_pos, start_pos + t, dtype=torch.long, device=device)

        # Forward embeddings: token + position
        tok_emb = self.transformer["wte"](idx)  # (B, T, d_model)
        pos_emb = self.transformer["wpe"](pos)  # (T, d_model)
        x = self.transformer["drop"](tok_emb + pos_emb)

        # Forward through Transformer blocks with layer-wise cache integration
        for i, block in enumerate(self.transformer["h"]):
            x = block(x, kv_cache=kv_cache, layer_idx=i)

        # Final LayerNorm
        x = self.transformer["ln_f"](x)

        # Project to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute cross-entropy loss if training targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return logits, loss

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> ForgeTransformer:
        """Instantiate model directly from a YAML config file."""
        config = ModelConfig.from_yaml(config_path)
        return cls(config)
