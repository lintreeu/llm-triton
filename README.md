# **Llama‑Triton**

> **Llama‑Triton** 以 **Python + Triton** 重寫 `llama.cpp` 推理核心，支援 FP16／BF16 推理、Page Attention、Rotary Embedding、多 GPU 分層，並且 **無需安裝 `transformers`** 即可載入 Hugging Face 檢查點。

---

## 目錄

* [功能特色](#功能特色)
* [安裝](#安裝)
* [權重格式與匯入](#權重格式與匯入)
* [快速上手](#快速上手)

  * [CLI](#cli)
  * [Python API](#python-api)
* [開發與測試](#開發與測試)

---

## 功能特色

| 功能                 | 說明                                                                    |
| ------------------ | --------------------------------------------------------------------- |
| **Triton Kernels** | GEMM、Page Attention、RMSNorm 皆採 Triton 自動調優。                           |
| **Page Attention** | 以「頁」為單位管理 KV Cache，長序列推理節省 GPU 記憶體。                                   |
| **極簡依賴**           | 僅需 `torch >= 2.3`、`triton == 2.2`，**不依賴 `transformers`**。             |
| **多 GPU 分層**       | `devices=["cuda:0","cuda:1"]` 即可將前 N 層切換至其他 GPU。                      |
| **權重雙格式**          | 原生 `state_dict.pt`＋`model.json`，或指定 `fmt="hf"` 直接匯入 Hugging Face 檢查點。 |
| **GGUF 轉換腳本**      | `scripts/convert_gguf.py` 可將 llama.cpp 權重轉為 PyTorch `state_dict.pt`。  |
| **完整測試**           | `pytest` 覆蓋率 > 90%，CI 於 GitHub Actions 自動執行。                          |

---

## 安裝

```bash
# 建議 Python >= 3.10，CUDA 12.x
conda create -n llama-triton python=3.10 -y
conda activate llama-triton

pip install -r requirements.txt  # 僅 torch + triton + sentencepiece + numpy

# （可選）若需 .safetensors 匯入 Hugging Face 權重
pip install safetensors
```

> **僅使用 CPU？** 於 `pip install` 加上 `--extra-index-url https://download.pytorch.org/whl/cpu`。目前 Triton kernel 不支援 CPU，會自動回退到 PyTorch 實作（效能較低）。

---

## 權重格式與匯入

### 1 · 原生 Llama‑Triton 權重

```
model_dir/
 ├─ model.json        # graph.config.LlamaConfig
 ├─ state_dict.pt     # PyTorch state_dict（鍵已對應 Triton 模組）
 └─ tokenizer.model   # SentencePiece
```

### 2 · Hugging Face 檢查點

將官方檔案下載到資料夾，例如：

```bash
huggingface-cli download twinkle-ai/Llama-3.2-3B-F1-Instruct --local-dir ./hf-ckpt
```

初始化 `LlamaGenerator(fmt="hf")` 會在本地呼叫 `utils.hf_import.convert()`，
把 `config.json` + `pytorch_model-*.bin / *.safetensors` 轉成 `model.json` + `state_dict.pt`，
並快取於同一資料夾。**過程全程離線、不需 `transformers`**。

> 若已手動執行 `python -m utils.hf_import ./hf-ckpt --dtype bfloat16` 可直接將 `fmt` 設為 `triton`。

---

## 快速上手

### CLI

```bash
# 原生權重
python main.py ./models/llama2-7b -p "台灣的首都是？" --max_tokens 16

# Hugging Face 權重
python main.py ./hf-ckpt -p "台灣的首都是？" --max_tokens 16 --fmt hf
```

終端輸出：

```text
› 台北市。
```

### Python API

```python
from api.inference import LlamaGenerator

# Triton 原生權重
gen = LlamaGenerator("./models/llama-3.2-3b", fmt="triton", device="cuda")
print(gen.generate("The capital of France is", max_new_tokens=16))

# Hugging Face 權重（自動轉換並快取）
gen_hf = LlamaGenerator("./hf-ckpt", fmt="hf", device="cuda", dtype="bfloat16")
print(gen_hf.generate("What is the tallest mountain?", max_new_tokens=16))
```

---

## 開發與測試

```bash
# 安裝開發依賴
pip install -r requirements-dev.txt

# 執行全部測試
pytest -q
```

---

## 授權

本專案採用 MIT License，詳細條款請見 `LICENSE` 檔案。
