"""Single transformer decoder block accelerated with Triton kernels."""

from __future__ import annotations

from typing import Tuple

import torch

from graph.config import LlamaConfig
from runtime.kv_cache import PageKVCache
from triton_kernels import rotary, rmsnorm
from triton_kernels.matmul import triton_gemm
from triton_kernels.page_attention import page_attention


class LlamaBlock(torch.nn.Module):
    """A single decoder block using Triton kernels."""

    def __init__(
        self,
        cfg: LlamaConfig,
        kv_cache: PageKVCache,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.kv_cache = kv_cache
        self.layer_idx = layer_idx

        dtype = getattr(torch, cfg.torch_dtype, torch.float16)
        init_std = cfg.initializer_range

        hidden = cfg.hidden_size
        interm = cfg.intermediate_size

        # ─────────────── Attention weights ─────────────── #
        self.w_q = torch.nn.Parameter(torch.randn(hidden, hidden, dtype=dtype) * init_std)
        self.w_k = torch.nn.Parameter(torch.randn(hidden, hidden, dtype=dtype) * init_std)
        self.w_v = torch.nn.Parameter(torch.randn(hidden, hidden, dtype=dtype) * init_std)
        self.w_o = torch.nn.Parameter(torch.randn(hidden, hidden, dtype=dtype) * init_std)
        self.attn_norm = torch.nn.Parameter(torch.ones(hidden, dtype=dtype))

        # ─────────────── Feed-forward weights ─────────────── #
        self.w_ff1 = torch.nn.Parameter(torch.randn(hidden, interm, dtype=dtype) * init_std)
        self.w_ff2 = torch.nn.Parameter(torch.randn(interm, hidden, dtype=dtype) * init_std)
        self.ffn_norm = torch.nn.Parameter(torch.ones(hidden, dtype=dtype))

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        pos: int,
        sin_cache: torch.Tensor,
        cos_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Run one decoder block on a single token."""
        h_attn = rmsnorm.rmsnorm(x, self.attn_norm)

        q = triton_gemm(h_attn[None, :], self.w_q).squeeze(0)
        k = triton_gemm(h_attn[None, :], self.w_k).squeeze(0)
        v = triton_gemm(h_attn[None, :], self.w_v).squeeze(0)

        n_heads = self.cfg.num_heads
        n_kv_heads = self.cfg.num_kv_heads
        head_dim = self.cfg.head_dim

        q = q.view(n_heads, head_dim)
        k = k.view(n_kv_heads, head_dim)
        v = v.view(n_kv_heads, head_dim)

        q = rotary.apply_rotary(q, sin_cache, cos_cache, pos)
        k = rotary.apply_rotary(k, sin_cache, cos_cache, pos)

        self.kv_cache.allocate(self.layer_idx, k, v)
        kv_pages: Tuple[torch.Tensor, torch.Tensor] = self.kv_cache.get_kv_pages(
            self.layer_idx, device=x.device
        )
        o_attn = page_attention(q, kv_pages)
        o_attn = triton_gemm(o_attn.view(-1)[None, :], self.w_o).squeeze(0)

        x = x + o_attn  # Residual

        # ─────────────── FFN ─────────────── #
        h_ffn = rmsnorm.rmsnorm(x, self.ffn_norm)
        ff_inter = triton_gemm(h_ffn[None, :], self.w_ff1).squeeze(0)
        ff_activated = torch.nn.functional.silu(ff_inter)
        ff_out = triton_gemm(ff_activated[None, :], self.w_ff2).squeeze(0)

        return x + ff_out  # Residual
