"""Paged key–value cache backing Triton page-attention kernels.

The cache organises (K, V) tensors into fixed-size pages so that attention
kernels can fetch contiguous memory blocks regardless of sequence length.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from graph.config import LlamaConfig


class PageKVCache:
    """Ring-buffer style KV cache with page granularity.

    Args:
        cfg: Model configuration holding cache hyperparameters.
        device: Torch device on which the cache tensors are allocated.
    """

    def __init__(self, cfg: LlamaConfig, device: str | torch.device = "cuda") -> None:
        self.cfg: LlamaConfig = cfg
        self.device: torch.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )

        num_kv_heads = cfg.n_kv_heads
        head_dim = cfg.hidden_size // cfg.n_heads

        # Shape: (layers, pages, page_size, kv_heads, head_dim)
        cache_shape = (
            cfg.n_layers,
            cfg.max_pages,
            cfg.page_size,
            num_kv_heads,
            head_dim,
        )
        dtype = getattr(torch, cfg.kv_dtype)

        self.k_cache = torch.empty(cache_shape, dtype=dtype, device=self.device)
        self.v_cache = torch.empty_like(self.k_cache)

        # Ring-buffer state.
        self.cur_page: int = 0     # Index of the page currently being written.
        self.cur_pos: int = 0      # Position inside the current page.
        self.total_tokens: int = 0

    # ------------------------------------------------------------------ #
    # Public helpers                                                     #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def allocate(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append one (k, v) pair to the cache for a given layer.

        If the current page is full, advance to the next page (ring-buffer
        semantics) and overwrite stale data.

        Args:
            layer: Decoder layer index.
            k: Key tensor of shape ``(kv_heads, head_dim)``.
            v: Value tensor with the same shape as ``k``.
        """
        if self.cur_pos >= self.cfg.page_size:
            self.cur_page = (self.cur_page + 1) % self.cfg.max_pages
            self.cur_pos = 0

        self.k_cache[layer, self.cur_page, self.cur_pos].copy_(k)
        self.v_cache[layer, self.cur_page, self.cur_pos].copy_(v)

        self.cur_pos += 1
        self.total_tokens += 1

    def get_kv_pages(
        self,
        layer: int,
        device: str | torch.device | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a slice of pages covering the entire context.

        Pages are ordered chronologically from oldest to newest.

        Args:
            layer: Decoder layer index.
            device: Target device for the returned tensors.  Defaults to the
                cache device.

        Returns:
            Tuple ``(k_pages, v_pages)`` each with shape
            ``(num_pages, page_size, kv_heads, head_dim)``.
        """
        target_device = device if device is not None else self.device

        num_pages = min(
            math.ceil(self.total_tokens / self.cfg.page_size),
            self.cfg.max_pages,
        )
        # Generate page indices in chronological order.
        order = [
            (self.cur_page - i) % self.cfg.max_pages for i in range(num_pages)
        ][::-1]

        k_pages = self.k_cache[layer, order].to(target_device)
        v_pages = self.v_cache[layer, order].to(target_device)
        return k_pages, v_pages
