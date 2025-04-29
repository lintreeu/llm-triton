"""Minimal Llama model rewritten for Triton kernels.

This module defines `LlamaTritonModel`, a lightweight, Triton-accelerated
decoder-only Transformer compatible with the *Llama* architecture.
"""

from __future__ import annotations

from typing import Optional

import torch

from graph.block import LlamaBlock
from graph.config import LlamaConfig
from runtime.kv_cache import PageKVCache
from triton_kernels.matmul import triton_gemm
from triton_kernels.rotary import build_rotary_cache
from triton_kernels import rmsnorm


class LlamaTritonModel(torch.nn.Module):
    """Decoder-only Transformer with Triton-based kernels."""

    def __init__(
        self,
        cfg: LlamaConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()

        self.cfg: LlamaConfig = cfg
        self.device: torch.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )

        # Page-wise KV cache shared by all attention blocks.
        self.kv_cache = PageKVCache(cfg, device=self.device)

        # Token embedding table.
        self.tok_embedding = torch.nn.Embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            dtype=torch.float16,
            device=self.device,
        )

        # Decoder blocks.
        self.blocks = torch.nn.ModuleList(
            [
                LlamaBlock(cfg, self.kv_cache, layer_idx=i).to(self.device)
                for i in range(cfg.num_layers)
            ]
        )

        # Output RMSNorm weight (γ).
        self.norm_out = torch.nn.Parameter(
            torch.ones(cfg.hidden_size, dtype=torch.float16, device=self.device)
        )

        # Output projection (tied or untied).
        if cfg.tie_weights:
            self.lm_head: torch.nn.Parameter | torch.Tensor = self.tok_embedding.weight
        else:
            self.lm_head = torch.nn.Parameter(
                torch.randn(
                    cfg.hidden_size,
                    cfg.vocab_size,
                    dtype=torch.float16,
                    device=self.device,
                )
                * 0.02
            )

        # Pre-computed rotary embeddings.
        sin, cos = build_rotary_cache(
            cfg.rope_freq_base,
            cfg.hidden_size // cfg.num_heads,
            cfg.max_position_embeddings,
            dtype=torch.float16,
        )
        self.sin_cache = sin.to(self.device)
        self.cos_cache = cos.to(self.device)

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of token IDs.

        Args:
            input_ids: Tensor of shape (batch, seq_len) or (seq_len,) with
                dtype = torch.long.

        Returns:
            Tensor of shape (batch, vocab_size) containing the logits of the
            final token in the sequence.
        """
        # Ensure a 2-D shape: (batch, seq_len).
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size, seq_len = input_ids.shape
        logits: Optional[torch.Tensor] = None

        for t in range(seq_len):
            # ------------------------------------------------------------------
            # 1. Token embedding lookup.
            # ------------------------------------------------------------------
            x = self.tok_embedding(input_ids[:, t])  # Shape: (batch, hidden)

            # Current absolute position (KV cache tracks global positions).
            pos = self.kv_cache.total_tokens or t

            # ------------------------------------------------------------------
            # 2. Decoder stack.
            # ------------------------------------------------------------------
            for block in self.blocks:
                x = block(
                    x.squeeze(0),  # Remove batch dim for per-token processing.
                    pos,
                    self.sin_cache,
                    self.cos_cache,
                ).unsqueeze(0)

            # ------------------------------------------------------------------
            # 3. Final RMSNorm + projection to vocab size.
            # ------------------------------------------------------------------
            x_norm = rmsnorm.rmsnorm(x.squeeze(0), self.norm_out)  # (hidden,)
            logits = triton_gemm(x_norm[None, :], self.lm_head).squeeze(0)

        return logits  # (vocab_size,)
