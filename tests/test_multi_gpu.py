import torch, pytest, os
from graph.config import LlamaConfig
from graph.model import LlamaTritonModel

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need ≥2 GPUs")
def test_multi_gpu_split():
    cfg = LlamaConfig(
        n_layers=4, n_heads=4, n_kv_heads=4,
        hidden_size=256, intermediate_size=512,
        vocab_size=32000, max_position_embeddings=128,
    )
    cfg.extra["devices"] = [f"cuda:{i}" for i in range(2)]
    cfg.extra["n_gpu_layers"] = 2
    model = LlamaTritonModel(cfg, device="cuda:1")
    mp = model.gpu_map
    assert mp.count("cuda:0") == 2 and mp.count("cuda:1") == 2
    del model
