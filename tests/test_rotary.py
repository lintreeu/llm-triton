import torch, pytest, math
from triton_kernels.rotary import build_rotary_cache, apply_rotary

@pytest.mark.parametrize("dim", [32, 64])
@pytest.mark.parametrize("pos", [0, 77, 511])
@pytest.mark.parametrize("device", ["cpu", "cuda"])   # ← 如果原專案 fixture 已提供，可刪
def test_rotary(dim, pos, device):
    base = 10000.0

    # 1. 先產生 cache（CPU），再搬到目標裝置
    sin, cos = build_rotary_cache(base, dim, pos + 4, torch.float32)
    sin, cos = sin.to(device), cos.to(device)

    # 2. 待測向量
    vec = torch.randn(dim, device=device, dtype=torch.float32)
    out = apply_rotary(vec, sin, cos, pos)

    # 3. 手動計算（與 vec 同裝置 / dtype）
    inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, device=device, dtype=vec.dtype) / dim)
    theta = pos * inv_freq

    rot = vec.clone()
    rot[0::2] = vec[0::2] * torch.cos(theta) - vec[1::2] * torch.sin(theta)
    rot[1::2] = vec[0::2] * torch.sin(theta) + vec[1::2] * torch.cos(theta)

    torch.testing.assert_close(out, rot, rtol=1e-4, atol=1e-4)
