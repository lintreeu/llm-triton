"""
utils.hf_import
===============

將 Hugging Face Llama-family 檢查點
(config.json + pytorch_model-*.bin / *.safetensors)
轉成 Llama-Triton 可用的：

    • model.json   （符合 graph.config.LlamaConfig）
    • state_dict.pt（對應 blocks.* 權重命名）

用法
----
$ python -m utils.hf_import ./hf-ckpt --dtype float16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

try:
    from safetensors.torch import load_file as safe_load
except ImportError:  # safetensors 為可選
    safe_load = None


# ────────────────────────────── 內部工具 ────────────────────────────── #
def _merge_weight_shards(files: List[Path]) -> Dict[str, torch.Tensor]:
    """讀入多個 shard，合併成單一 state_dict。"""
    sd: Dict[str, torch.Tensor] = {}
    for f in files:
        if f.suffix == ".bin":
            sd.update(torch.load(f, map_location="cpu"))
        elif f.suffix == ".safetensors":
            if safe_load is None:
                raise RuntimeError("檢測到 .safetensors，但未安裝 safetensors 套件。")
            sd.update(safe_load(str(f)))
        else:
            raise ValueError(f"不支援的權重檔案格式：{f}")
    return sd


def _rename_keys(hf_sd: Dict[str, torch.Tensor], n_layers: int) -> Dict[str, torch.Tensor]:
    """將 Hugging Face 權重鍵改成 Llama-Triton 名稱。"""
    out: Dict[str, torch.Tensor] = {}

    # 嵌入 / LM 頭
    out["tok_embedding.weight"] = hf_sd.pop("model.embed_tokens.weight")

    # 各層
    for i in range(n_layers):
        prefix = f"model.layers.{i}."

        out[f"blocks.{i}.w_q"] = hf_sd.pop(prefix + "self_attn.q_proj.weight")
        out[f"blocks.{i}.w_k"] = hf_sd.pop(prefix + "self_attn.k_proj.weight")
        out[f"blocks.{i}.w_v"] = hf_sd.pop(prefix + "self_attn.v_proj.weight")
        out[f"blocks.{i}.w_o"] = hf_sd.pop(prefix + "self_attn.o_proj.weight")

        out[f"blocks.{i}.attn_norm"] = hf_sd.pop(prefix + "input_layernorm.weight")
        out[f"blocks.{i}.ffn_norm"] = hf_sd.pop(prefix + "post_attention_layernorm.weight")

        # 簡化 FFN：使用 up_proj 與 down_proj
        out[f"blocks.{i}.w_ff1"] = hf_sd.pop(prefix + "mlp.up_proj.weight")
        out[f"blocks.{i}.w_ff2"] = hf_sd.pop(prefix + "mlp.down_proj.weight")
        # gate_proj 在本實作被省略，釋放記憶體
        _ = hf_sd.pop(prefix + "mlp.gate_proj.weight")

    # 最終 RMSNorm
    out["norm_out"] = hf_sd.pop("model.norm.weight")

    # LM head（若未綁定可額外保存）
    if "lm_head.weight" in hf_sd:
        out["lm_head"] = hf_sd.pop("lm_head.weight")

    return out


# ────────────────────────────── 主要流程 ────────────────────────────── #
def convert(model_dir: Path, target_dtype: str = "float16") -> None:
    """將 HF 檔案轉成 Llama-Triton 需要的 model.json + state_dict.pt。"""
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("未找到 config.json，無法辨識為 HF 檔案資料夾。")

    # 讀取並轉成本專案 LlamaConfig
    from graph.config import LlamaConfig  # 避免循環相依

    llama_cfg = LlamaConfig.from_json(cfg_path)
    llama_cfg.torch_dtype = target_dtype
    # 輸出 model.json
    (model_dir / "model.json").write_text(
        json.dumps(llama_cfg.__dict__, indent=2, ensure_ascii=False)
    )

    # 讀取權重 shard
    weight_files = sorted(
        list(model_dir.glob("pytorch_model*.bin")) + list(model_dir.glob("*.safetensors"))
    )
    if not weight_files:
        raise FileNotFoundError("未找到 *.bin 或 *.safetensors 權重檔案。")

    hf_state = _merge_weight_shards(weight_files)
    state_dict = _rename_keys(hf_state, llama_cfg.num_layers)

    # dtype 轉換
    torch_dtype = getattr(torch, target_dtype)
    for k in state_dict:
        state_dict[k] = state_dict[k].to(torch_dtype)

    torch.save(state_dict, model_dir / "state_dict.pt")
    print(f"✓ 已輸出 {model_dir/'state_dict.pt'} 與 {model_dir/'model.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HF → Llama-Triton 轉換器")
    parser.add_argument("model_dir", help="Hugging Face checkpoint 資料夾")
    parser.add_argument("--dtype", default="float16", help="float16 | bfloat16 ...")
    args = parser.parse_args()
    convert(Path(args.model_dir), args.dtype)


if __name__ == "__main__":
    main()
