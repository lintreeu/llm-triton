"""Triton‐based GEMM (matrix multiplication)

The kernel computes C = A @ B on GPU using a tiled algorithm.  Autotuning
selects the best block configuration for a given (M, N, K) workload.

Functions
---------
matmul_kernel : Triton JIT kernel.
triton_gemm   : Python wrapper that dispatches the kernel on PyTorch tensors.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

###############################################################################
#                            Triton JIT kernel                                #
###############################################################################


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=2,
        )
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(  
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sa0,
    sa1,
    sb0,
    sb1,
    sc0,
    sc1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Tiled GEMM kernel executed by each CTA (Co-operative Thread Array).

    Each CTA computes a (BLOCK_M × BLOCK_N) tile of the output matrix C.
    """
    # ----------------------------------------------------------------------------
    # Identify the tile this CTA is responsible for (row_idx, col_idx).
    # ----------------------------------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    row_idx = pid // num_pid_m
    col_idx = pid % num_pid_m

    # Tile-local indices along M, N, and K dimensions.
    offs_m = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = col_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers for the current K slice of tiles from A and B.
    a_ptrs = a_ptr + offs_m[:, None] * sa0 + offs_k[None, :] * sa1
    b_ptrs = b_ptr + offs_k[:, None] * sb0 + offs_n[None, :] * sb1

    # Accumulator for the partial result (FP32 to reduce numerical error).
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ----------------------------------------------------------------------------
    # Loop over K dimension in BLOCK_K chunks.
    # ----------------------------------------------------------------------------
    for k in range(0, K, BLOCK_K):
        # Boundary masks to guard out-of-range loads.
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        mask_b = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)

        # Load sub-tiles of A and B; fill OOB elements with zeros.
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # FMA accumulation: acc += A_tile @ B_tile
        acc += tl.dot(a, b)

        # Advance pointers to the next K slice.
        a_ptrs += BLOCK_K * sa1
        b_ptrs += BLOCK_K * sb0

    # ----------------------------------------------------------------------------
    # Write the accumulator back to C with boundary masking.
    # ----------------------------------------------------------------------------
    c_ptrs = c_ptr + offs_m[:, None] * sc0 + offs_n[None, :] * sc1
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)

###############################################################################
#                        Public Python-level wrapper                           #
###############################################################################


def triton_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute a @ b using the Triton kernel.

    Parameters
    ----------
    a : torch.Tensor
        Input matrix A with shape (M, K).
    b : torch.Tensor
        Input matrix B with shape (K, N).

    Returns
    -------
    torch.Tensor
        Resulting matrix C with shape (M, N) and dtype matching ``a``.
    """
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError('A.shape[1] must equal B.shape[0]')

    # Use FP32 for accumulation even if inputs are lower precision.
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    def grid(meta: dict[str, int]) -> Tuple[int]:
        """Compute total number of CTAs for the given meta parameters."""
        return (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        )

    # Launch the kernel.
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )

    # Cast back to the original dtype (e.g., fp16 or bf16).
    return c.to(dtype=a.dtype)
