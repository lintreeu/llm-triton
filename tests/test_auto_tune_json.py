from pathlib import Path
import json, pytest

def test_auto_json():
    path = Path("triton_kernels/page_attn_auto.json")
    assert path.exists(), "auto-tune json not found"
    data = json.loads(path.read_text())
    assert all(isinstance(k, str) for k in data.keys())
