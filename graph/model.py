"""Decoder-only Llama model rewritten with Triton kernels."""

from __future__ import annotations

from typing import Optional

import torch

from graph.block import LlamaBlock
from graph.config import LlamaConfig
from runtime.kv_cache import PageKVCache
from triton_kernels import rmsnorm, rotary
from triton_kernels.matmul import triton_gemm


class LlamaTritonModel(torch.nn.Module):
    """Llama-style decoder-only Transformer (Triton kernels)."""

    def __init__(
        self,
        cfg: LlamaConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)

        dtype = getattr(torch, cfg.torch_dtype, torch.float16)

        # --- 一致性檢查 --- #
        expected_hidden = cfg.num_heads * cfg.head_dim
        if expected_hidden != cfg.hidden_size:
            raise ValueError(
                f"hidden_size 應為 num_heads × head_dim = {expected_hidden}, "
                f"但得到 {cfg.hidden_size}"
            )

        # KV cache
        self.kv_cache = PageKVCache(cfg, device=self.device)

        # Token embedding
        self.tok_embedding = torch.nn.Embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            dtype=dtype,
            device=self.device,
        )

        # Decoder blocks
        self.blocks = torch.nn.ModuleList(
            [
                LlamaBlock(cfg, self.kv_cache, i).to(self.device)
                for i in range(cfg.num_layers)
            ]
        )

        # Output RMSNorm
        self.norm_out = torch.nn.Parameter(
            torch.ones(cfg.hidden_size, dtype=dtype, device=self.device)
        )

        # LM head
        if cfg.tie_weights:
            self.lm_head = self.tok_embedding.weight
        else:
            self.lm_head = torch.nn.Parameter(
                torch.randn(
                    cfg.hidden_size,
                    cfg.vocab_size,
                    dtype=dtype,
                    device=self.device,
                )
                * cfg.initializer_range
            )

        # Rotary caches
        sin, cos = rotary.build_rotary_cache(
            cfg.rope_theta,
            cfg.head_dim,
            cfg.max_position_embeddings,
            rope_scaling=cfg.rope_scaling,
            dtype=dtype,
        )
        self.sin_cache = sin.to(self.device)
        self.cos_cache = cos.to(self.device)

    # ------------------------------------------------------------------ #
    # Forward (single-token streaming or full-seq)                       #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits of last token. `input_ids` shape: (batch, seq_len)."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        batch, seq_len = input_ids.shape
        logits: Optional[torch.Tensor] = None

        for t in range(seq_len):
            # 1. Embedding lookup
            x = self.tok_embedding(input_ids[:, t])
            pos = self.kv_cache.total_tokens or t

            # 2. Decoder stack (per-token, loop over layers)
            for block in self.blocks:
                x = block(
                    x.squeeze(0), pos, self.sin_cache, self.cos_cache
                ).unsqueeze(0)

            # 3. Final RMSNorm + LM head
            x_norm = rmsnorm.rmsnorm(x.squeeze(0), self.norm_out)
            logits = triton_gemm(x_norm[None, :], self.lm_head).squeeze(0)

        return logits  # (vocab_size,)
