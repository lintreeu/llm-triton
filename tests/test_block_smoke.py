import torch
from graph.config import LlamaConfig
from runtime.kv_cache import PageKVCache
from graph.block import LlamaBlock
from triton_kernels.rotary import build_rotary_cache
def test_block(device):
    cfg=LlamaConfig(n_layers=1,n_heads=2,n_kv_heads=2,hidden_size=32,intermediate_size=64,
                    vocab_size=32,max_position_embeddings=64)
    kv=PageKVCache(cfg,device=device)
    blk=LlamaBlock(cfg,kv,0).to(device)
    sin,cos=build_rotary_cache(10000,16,64,torch.float16)
    x=torch.randn(cfg.hidden_size,dtype=torch.float16,device=device)
    out=blk(x,0,sin.to(device),cos.to(device))
    assert out.shape==(cfg.hidden_size,)
