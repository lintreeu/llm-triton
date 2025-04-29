import torch,pytest
from triton_kernels.rmsnorm import rmsnorm
@pytest.mark.parametrize('hidden',[16,128])
def test_rmsnorm(hidden,device):
    x=torch.randn(hidden,dtype=torch.float16,device=device)
    w=torch.randn_like(x)
    y=rmsnorm(x,w)
    rms=((x.float()*x.float()).mean())**0.5
    ref=(x/rms)*w
    torch.testing.assert_close(y,ref,rtol=3e-3,atol=3e-3)
