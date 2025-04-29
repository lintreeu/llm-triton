import torch,pytest,math
from triton_kernels.page_attention import page_attention
from tests.helpers import baseline_page_attention
@pytest.mark.parametrize('dims',[(2,8,4,32)])
def test_page_attn(dims,device):
    P,L,H,D=dims
    if device=='cpu': pytest.skip()
    q=torch.randn(H,D,dtype=torch.float16,device=device)
    k=torch.randn(P,L,H,D,dtype=torch.float16,device=device)
    v=torch.randn_like(k)
    tri=page_attention(q,(k,v)).float()
    ref=baseline_page_attention(q.float(),k.float(),v.float())
    torch.testing.assert_close(tri,ref,rtol=2e-2,atol=2e-2)
