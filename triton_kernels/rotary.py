# rotary.py
from __future__ import annotations
from typing import Tuple
import torch


def build_rotary_cache(
    base: float,
    dim: int,
    max_pos: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if dim % 2:
        raise ValueError("`dim` 必須為偶數")

    device = torch.device(device) if device is not None else torch.device("cpu")

    # 1) 角度 θ = pos / base^(2i/dim)
    inv_freq = torch.pow(
        base,
        -torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim,
    )                      # (dim//2,)
    positions = torch.arange(max_pos, device=device)[:, None]
    angle = positions * inv_freq                       # (max_pos, dim//2)

    # 2) sin / cos，並讓每個頻率在 even / odd 位置連續出現
    sin = torch.sin(angle)            # (max_pos, dim//2)
    cos = torch.cos(angle)            # (max_pos, dim//2)
    sin = sin.repeat_interleave(2, dim=1).to(dtype)   # (max_pos, dim)
    cos = cos.repeat_interleave(2, dim=1).to(dtype)

    return sin, cos


def apply_rotary(
    t: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    pos: int,
) -> torch.Tensor:
    if pos >= sin.size(0):
        raise IndexError(f"pos {pos} 超過 cache 長度 {sin.size(0)}")

    s = sin[pos].to(dtype=t.dtype, device=t.device)    # (dim,)
    c = cos[pos].to(dtype=t.dtype, device=t.device)

    t_even, t_odd = t[..., 0::2], t[..., 1::2]         # (..., dim/2)

    out_even = t_even * c[0::2] - t_odd * s[0::2]
    out_odd  = t_odd  * c[0::2] + t_even * s[0::2]

    out = torch.empty_like(t)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out
