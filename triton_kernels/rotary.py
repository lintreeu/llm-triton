# rotary.py
"""Rotary positional embedding utilities with Llama-3 rope_scaling support."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def _maybe_apply_scaling(
    max_pos: int,
    base: float,
    rope_scaling: Optional[Dict[str, float]],
) -> Tuple[int, float]:
    """處理 Llama-3 rope_scaling 規格 (factor, base)。"""
    if rope_scaling is None:
        return max_pos, base

    factor = float(rope_scaling.get("factor", 1.0))
    new_max_pos = int(max_pos * factor)
    # HF 把原始 theta 放在 rope_theta；factor > 1 代表外插長序列
    return new_max_pos, base


def build_rotary_cache(
    base: float,
    dim: int,
    max_pos: int,
    *,
    rope_scaling: Optional[Dict[str, float]] = None,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (sin, cos) lookup tables of shape (max_pos, dim)."""
    if dim % 2:
        raise ValueError("`dim` 必須為偶數")

    max_pos, base = _maybe_apply_scaling(max_pos, base, rope_scaling)
    device = torch.device(device) if device is not None else torch.device("cpu")

    inv_freq = torch.pow(
        base,
        -torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim,
    )  # (dim//2,)

    positions = torch.arange(max_pos, device=device)[:, None]
    angle = positions * inv_freq  # (max_pos, dim//2)

    sin = torch.sin(angle).repeat_interleave(2, dim=1).to(dtype)
    cos = torch.cos(angle).repeat_interleave(2, dim=1).to(dtype)
    return sin, cos


def apply_rotary(
    t: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    pos: int,
) -> torch.Tensor:
    """Apply rotary embedding to the last dimension of `t` (out-of-place)."""
    if pos >= sin.size(0):
        raise IndexError(f"pos {pos} 超出 cache 長度 {sin.size(0)}")

    s = sin[pos].to(dtype=t.dtype, device=t.device)
    c = cos[pos].to(dtype=t.dtype, device=t.device)

    t_even, t_odd = t[..., 0::2], t[..., 1::2]
    out_even = t_even * c[0::2] - t_odd * s[0::2]
    out_odd = t_odd * c[0::2] + t_even * s[0::2]

    out = torch.empty_like(t)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out
