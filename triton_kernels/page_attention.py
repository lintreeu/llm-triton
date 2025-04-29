"""Page-wise scaled dot-product attention implemented with Triton.

The kernel supports a “paged” KV cache layout:

    * num_pages (P)  – number of pages in the cache
    * seq_len  (L)   – tokens per page
    * num_heads (H)  – attention heads
    * head_dim (D)   – embedding dimension per head

Shape conventions
-----------------
q            : (H, D)
k, v         : (P, L, H, D)
o (output)   : (H, D)

Public API
----------
page_attention(q, (k, v)) -> o
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

###############################################################################
#                               Triton kernel                                 #
###############################################################################


@triton.jit
def _page_attn_kernel( 
    q_ptr,  # Query pointer
    k_ptr,  # Key pointer
    v_ptr,  # Value pointer
    o_ptr,  # Output pointer
    num_pages: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_q_head,
    stride_q_dim,
    stride_k_page,
    stride_k_seq,
    stride_k_head,
    stride_k_dim,
    stride_v_page,
    stride_v_seq,
    stride_v_head,
    stride_v_dim,
    stride_o_head,
    stride_o_dim,
    BLOCK_D: tl.constexpr,
) -> None:
    """Compute one head of page attention for a `BLOCK_D` slice of `head_dim`."""

    # ------------------------------------------------------------------ #
    # Locate the current CTA (head_id, dim_block_id)                     #
    # ------------------------------------------------------------------ #
    pid = tl.program_id(axis=0)
    num_dim_blocks = tl.cdiv(head_dim, BLOCK_D)

    head_id = pid // num_dim_blocks          # Which attention head
    dim_blk_id = pid % num_dim_blocks        # Which dim slice inside the head

    offs_d = dim_blk_id * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # ------------------------------------------------------------------ #
    # Load query slice Q[h, offs_d]                                       #
    # ------------------------------------------------------------------ #
    q_ptrs = q_ptr + head_id * stride_q_head + offs_d
    q = tl.load(q_ptrs, mask=d_mask, other=0.0)

    # Running mean / normalizer for numerically stable softmax.
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m_prev = tl.full((1,), float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros((1,), dtype=tl.float32)

    inv_sqrt_d = 1.0 / tl.sqrt(tl.float32(head_dim))

    # ------------------------------------------------------------------ #
    # Iterate over pages and tokens                                      #
    # ------------------------------------------------------------------ #
    for p in range(num_pages):
        for t in range(seq_len):
            k_ptrs = (
                k_ptr
                + p * stride_k_page
                + t * stride_k_seq
                + head_id * stride_k_head
                + offs_d
            )
            v_ptrs = (
                v_ptr
                + p * stride_v_page
                + t * stride_v_seq
                + head_id * stride_v_head
                + offs_d
            )

            k = tl.load(k_ptrs, mask=d_mask, other=0.0)
            v = tl.load(v_ptrs, mask=d_mask, other=0.0)

            # Scaled dot-product (scalar score for this (head, token))
            qk = tl.sum(q * k, axis=0, output_dtype=tl.float32) * inv_sqrt_d

            # Online softmax update (see “Stable Softmax” trick)
            m_curr = tl.maximum(m_prev, qk)
            l_prev *= tl.exp(m_prev - m_curr)
            p_attn = tl.exp(qk - m_curr)
            l_curr = l_prev + p_attn

            acc = acc * (l_prev / l_curr) + v * (p_attn / l_curr)

            m_prev = m_curr
            l_prev = l_curr

    # ------------------------------------------------------------------ #
    # Store the output slice O[h, offs_d]                                 #
    # ------------------------------------------------------------------ #
    o_ptrs = o_ptr + head_id * stride_o_head + offs_d
    tl.store(o_ptrs, acc.to(q_ptr.dtype.element_ty), mask=d_mask)

###############################################################################
#                             Python wrapper                                  #
###############################################################################


def page_attention(
    q: torch.Tensor,
    kv_pages: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Page attention forward pass performed on GPU with Triton.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape (H, D).
    kv_pages : Tuple[torch.Tensor, torch.Tensor]
        Key / Value tensors, each of shape (P, L, H, D).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (H, D) with the same dtype / device as `q`.
    """
    k, v = kv_pages
    num_pages, seq_len, num_heads, head_dim = k.shape

    # Sanity checks
    if q.shape != (num_heads, head_dim):
        raise ValueError('q must have shape (H, D) matching k / v.')

    o = torch.empty_like(q)

    BLOCK_D = 32  # Must divide warp size; tune for your GPU

    grid = (num_heads * triton.cdiv(head_dim, BLOCK_D),)

    _page_attn_kernel[grid](
        q,
        k,
        v,
        o,
        num_pages,
        seq_len,
        num_heads,
        head_dim,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        BLOCK_D,
    )

    return o
