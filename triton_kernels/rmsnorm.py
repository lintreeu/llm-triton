"""
Rotary positional embedding utilities.

Functions
---------
1. build_rotary_cache : Pre-compute sin / cos tables.
2. apply_rotary       : Apply rotary embedding to query / key tensors.

Code follows PEP 8 & Google Python Style Guide, comments in English.
"""

from __future__ import annotations

from typing import Tuple

import torch


# --------------------------------------------------------------------------- #
# Cache builder
# --------------------------------------------------------------------------- #
def build_rotary_cache(
    base: float,
    dim: int,
    max_pos: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (sin, cos) lookup tables of shape ``(max_pos, dim)``."""
    if dim % 2:
        raise ValueError("`dim` must be even for rotary embedding.")
    print(device)
    device = torch.device(device) if device is not None else torch.device("cpu")

    inv_freq = torch.pow(
        base,
        -torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim,
    )  # (dim/2,)

    positions = torch.arange(max_pos, device=device)[:, None]      # (max_pos, 1)
    freqs = positions * inv_freq                                   # (max_pos, dim/2)
    emb = torch.cat([freqs, freqs], dim=1)                         # (max_pos, dim)

    sin = torch.sin(emb, dtype=dtype)
    cos = torch.cos(emb, dtype=dtype)
    return sin, cos


# --------------------------------------------------------------------------- #
# Rotary application
# --------------------------------------------------------------------------- #
def apply_rotary(
    t: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    pos: int,
) -> torch.Tensor:
    """Apply rotary embedding to the *last* dimension of ``t`` (out-of-place)."""
    if pos >= sin.size(0):
        raise IndexError(f"pos {pos} exceeds cache size {sin.size(0)}")

    # --- 保證後續新建 Tensor 與輸入位於同一裝置 (for unit-test) ---
    if t.device.type == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    # -----------------------------------------------------------------------

    # Prepare sin / cos for broadcasting
    s = sin[pos].to(dtype=t.dtype, device=t.device)    # (dim,)
    c = cos[pos].to(dtype=t.dtype, device=t.device)    # (dim,)
    s_half, c_half = s[0::2], c[0::2]                  # (dim/2,)

    # Split even / odd channels
    t_even = t[..., 0::2]                              # (..., dim/2)
    t_odd  = t[..., 1::2]                              # (..., dim/2)

    # (x, y) → (x · cos − y · sin, y · cos + x · sin)
    out_even = t_even * c_half - t_odd * s_half
    out_odd  = t_odd  * c_half + t_even * s_half

    # Interleave even / odd back
    out = torch.empty_like(t)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out
