import torch, pytest
from triton_kernels.matmul import triton_gemm
@pytest.mark.parametrize('dtype',[torch.float16])
@pytest.mark.parametrize('shape',[(16,16,16),(32,64,32)])
def test_matmul(shape,dtype,device):
    M,N,K=shape
    a=torch.randn(M,K,device=device,dtype=dtype)
    b=torch.randn(K,N,device=device,dtype=dtype)
    ref=(a.float()@b.float()).to(dtype)
    out=triton_gemm(a,b)
    torch.testing.assert_close(out,ref,rtol=5e-3,atol=5e-3)
