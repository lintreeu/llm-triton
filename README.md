
# **Llama-Triton** 


> **Llama-Triton** 是將 `llama.cpp` 核心以 **Python + Triton** 重寫的專案，支援 FP16 推理、Page Attention、Rotary Embedding 與多 GPU 分層；容易 hack、易於學習，適合想深入研究 LLM 推理核心或客製化 GPU kernels 的開發者。

---

## 目錄

- [功能說明](#功能說明)
- [快速安裝](#快速安裝)
- [基本使用](#基本使用)
  - [CLI](#cli)
  - [Python API](#python-api)


---

## 功能說明

| 功能 | 說明 |
|---------|------|
| **Triton Kernels** | GEMM、Page Attention、RMSNorm 全以 Triton 撰寫，一鍵自動調優。 |
| **Page Attention** | 以「頁」為單位管理 KV-Cache，長序列推理顯著節省 GPU 記憶體。 |
| **極簡依賴** | 只需 `torch >= 2.3` 與 `triton == 2.2`。無需編譯 C++。 |
| **多 GPU 分層** | `devices=["cuda:0","cuda:1"]` 即可把前 N 層切到其它 GPU。 |
| **GGUF 轉換腳本** | `scripts/convert_gguf.py` 一鍵把 llama.cpp 權重轉成 PyTorch `state_dict.pt`。 |
| **完整測試覆蓋** | pytest 覆蓋率 > 90%，CI free on GitHub Actions。 |


## 快速安裝

```bash
# 建議 Python >= 3.10，CUDA 12.x
conda create -n llama-triton python=3.10 -y
conda activate llama-triton

pip install -r requirements.txt
```

> **若僅使用 CPU** 可加上 `--extra-index-url https://download.pytorch.org/whl/cpu`。不過目前 Triton kernel 需 CUDA 👉 CPU fallback 開發中。

## 基本使用

### CLI

```bash
python main.py ./models/llama2-7b \
  --prompt "台灣的首都是？" \
  --max_tokens 16
```

終端輸出：

```text
› 台北市。
```

### Python API

```python
from api.inference import LlamaGenerator

gen = LlamaGenerator("./models/llama2-7b", device="cuda")
print(gen.generate("The capital of France is", max_new_tokens=16))
```

> 亦可使用 `gen.generate_batch([...])` 進行批次推理。



