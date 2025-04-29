# tests/conftest.py
import os, random, pytest, torch

def pytest_configure(config):
    # 減少雜訊 & 保證 determinism
    torch.manual_seed(42)
    random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)

@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
