"""Minimal inference engine to validate Triton page attention and KV cache.

This class is **not** a full Llama model—just a stripped-down forward loop
for debugging Triton kernels.  It will eventually be superseded by
`graph.model.LlamaTritonModel`.
"""

from __future__ import annotations

from typing import Tuple

import torch

from graph.config import LlamaConfig
from runtime.kv_cache import PageKVCache
from triton_kernels.matmul import triton_gemm
from triton_kernels.page_attention import page_attention


class InferenceEngine:
    """Single-token forward pass with Triton-accelerated primitives."""

    def __init__(self, config: LlamaConfig, device: str | torch.device = "cuda") -> None:
        self.cfg: LlamaConfig = config
        self.device: torch.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )

        # Shared paged KV cache.
        self.kv_cache = PageKVCache(config, device=self.device)

        # ------------------------------------------------------------------ #
        # Lightweight random projections for q, k, v, o.                     #
        # ------------------------------------------------------------------ #
        init_std = 0.01
        hidden = config.hidden_size
        dtype = torch.float16

        self.dummy_wq = torch.randn(hidden, hidden, device=self.device, dtype=dtype) * init_std
        self.dummy_wk = torch.randn(hidden, hidden, device=self.device, dtype=dtype) * init_std
        self.dummy_wv = torch.randn(hidden, hidden, device=self.device, dtype=dtype) * init_std
        self.dummy_wo = torch.randn(hidden, hidden, device=self.device, dtype=dtype) * init_std

    # ---------------------------------------------------------------------- #
    # Public API                                                             #
    # ---------------------------------------------------------------------- #

    def step(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """Run a single-layer, single-token forward pass.

        Args:
            x: Activation vector of shape ``(hidden_size,)``.
            layer_idx: Which layer of the KV cache to append to.

        Returns:
            Updated activation vector of shape ``(hidden_size,)``.
        """
        # Cast to FP16 for Triton kernels.
        h = x.to(torch.float16)

        # 1. Linear projections via Triton GEMM.
        q = triton_gemm(h[None, :], self.dummy_wq).squeeze(0)
        k = triton_gemm(h[None, :], self.dummy_wk).squeeze(0)
        v = triton_gemm(h[None, :], self.dummy_wv).squeeze(0)

        # 2. Reshape into (num_heads, head_dim).
        num_heads = self.cfg.n_heads
        head_dim = self.cfg.hidden_size // num_heads

        q = q.view(num_heads, head_dim)
        k = k.view(self.cfg.n_kv_heads, head_dim)
        v = v.view(self.cfg.n_kv_heads, head_dim)

        # 3. Append K/V to the paged cache.
        self.kv_cache.allocate(layer_idx, k, v)

        # 4. Page attention (Triton kernel).
        kv_pages: Tuple[torch.Tensor, torch.Tensor] = self.kv_cache.get_kv_pages(layer_idx)
        attn_out = page_attention(q, kv_pages)  # Shape: (num_heads, head_dim)

        # 5. Output projection back to model dimension.
        y = triton_gemm(attn_out.reshape(-1)[None, :], self.dummy_wo).squeeze(0)
        return y
