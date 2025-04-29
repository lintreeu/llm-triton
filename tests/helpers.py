# tests/helpers.py
import torch
import math

def baseline_page_attention(q, k, v):
    """
    q: [H,D] , k/v: [P,L,H,D] (same H)
    returns: [H,D]  (fp32)
    """
    P, L, H, D = k.shape
    k_flat = k.reshape(P * L, H, D).transpose(0, 1)      # [H, S, D]
    v_flat = v.reshape(P * L, H, D).transpose(0, 1)      # [H, S, D]
    logits = torch.einsum("hd,hSd->hS", q, k_flat) / math.sqrt(D)
    weights = torch.softmax(logits, dim=-1)
    out = torch.einsum("hS,hSd->hd", weights, v_flat)
    return out
