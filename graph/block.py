"""Single transformer decoder block optimized with Triton kernels.

The block follows the Llama architecture:

    1. Pre-norm self-attention with rotary embeddings and a paged KV cache.
    2. Feed-forward network (SiLU/GEGLU style) with pre-norm.
    3. Residual connections after both attention and feed-forward sub-layers.

The implementation uses Triton GEMM / RMSNorm / attention kernels for speed.
"""

from __future__ import annotations

from typing import Tuple

import torch

from graph.config import LlamaConfig
from runtime.kv_cache import PageKVCache
from triton_kernels import rmsnorm, rotary
from triton_kernels.matmul import triton_gemm
from triton_kernels.page_attention import page_attention


class LlamaBlock(torch.nn.Module):
    """A single decoder block with Triton-accelerated primitives."""

    def __init__(
        self,
        cfg: LlamaConfig,
        kv_cache: PageKVCache,
        layer_idx: int,
    ) -> None:
        super().__init__()

        self.cfg: LlamaConfig = cfg
        self.kv_cache: PageKVCache = kv_cache
        self.layer_idx: int = layer_idx

        hidden_size = cfg.hidden_size
        interm_size = cfg.intermediate_size
        dtype = torch.float16

        # ────────────────────────── Attention weights ───────────────────────── #
        init_std = 0.02
        self.w_q = torch.nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype) * init_std)
        self.w_k = torch.nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype) * init_std)
        self.w_v = torch.nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype) * init_std)
        self.w_o = torch.nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype) * init_std)

        # Layer-norm scale (γ) before attention.
        self.attn_norm = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))

        # ────────────────────────── Feed-forward weights ────────────────────── #
        self.w_ff1 = torch.nn.Parameter(torch.randn(hidden_size, interm_size, dtype=dtype) * init_std)
        self.w_ff2 = torch.nn.Parameter(torch.randn(interm_size, hidden_size, dtype=dtype) * init_std)

        # Layer-norm scale (γ) before FFN.
        self.ffn_norm = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    # --------------------------------------------------------------------- #
    # Forward pass                                                         #
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,  # Shape: (hidden_size,)
        pos: int,
        sin_cache: torch.Tensor,
        cos_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Run a single transformer block.

        Args:
            x: Activation vector for the current token, shape ``(hidden_size,)``.
            pos: Absolute position in the sequence (for rotary embeddings).
            sin_cache: Pre-computed sin table (rotary).
            cos_cache: Pre-computed cos table (rotary).

        Returns:
            Updated activation vector of shape ``(hidden_size,)``.
        """
        # ------------------------------------------------------------------ #
        # 1. Self-attention sub-layer (pre-norm)                             #
        # ------------------------------------------------------------------ #
        h_attn = rmsnorm.rmsnorm(x, self.attn_norm)

        q = triton_gemm(h_attn[None, :], self.w_q).squeeze(0)
        k = triton_gemm(h_attn[None, :], self.w_k).squeeze(0)
        v = triton_gemm(h_attn[None, :], self.w_v).squeeze(0)

        num_heads = self.cfg.num_heads
        num_kv_heads = self.cfg.num_kv_heads
        head_dim = self.cfg.hidden_size // num_heads

        q = q.view(num_heads, head_dim)
        k = k.view(num_kv_heads, head_dim)
        v = v.view(num_kv_heads, head_dim)

        q = rotary.apply_rotary(q, sin_cache, cos_cache, pos)
        k = rotary.apply_rotary(k, sin_cache, cos_cache, pos)

        # Store (k, v) into the shared paged KV cache.
        self.kv_cache.allocate(self.layer_idx, k, v)

        # Retrieve the pages relevant for this head and run Triton attention.
        kv_pages: Tuple[torch.Tensor, torch.Tensor] = self.kv_cache.get_kv_pages(
            self.layer_idx,
            device=x.device,
        )
        o_attn = page_attention(q, kv_pages)                    # (num_heads, head_dim)
        o_attn = triton_gemm(o_attn.view(-1)[None, :], self.w_o).squeeze(0)

        x = x + o_attn  # Residual connection.

        # ------------------------------------------------------------------ #
        # 2. Feed-forward sub-layer (pre-norm)                               #
        # ------------------------------------------------------------------ #
        h_ffn = rmsnorm.rmsnorm(x, self.ffn_norm)

        ff_intermediate = triton_gemm(h_ffn[None, :], self.w_ff1).squeeze(0)
        ff_activated = torch.nn.functional.silu(ff_intermediate)
        ff_output = triton_gemm(ff_activated[None, :], self.w_ff2).squeeze(0)

        return x + ff_output  # Final residual connection.
