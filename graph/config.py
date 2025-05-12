"""Llama configuration dataclass.

* 兼容 Hugging Face Llama-family JSON（含 Llama-3 rope_scaling 等欄位）。
* 以 `from_json()` → `LlamaConfig` → 其他模組的資料流為核心。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class LlamaConfig:
    # ─────────────────────────── 基本尺寸 ─────────────────────────── #
    num_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int

    # ─────────────────────────── Rotary 相關 ─────────────────────────── #
    rope_theta: float = 10_000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    head_dim: Optional[int] = None          # 若為 None，於載入時計算

    # ─────────────────────────── 其他超參 ─────────────────────────── #
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    mlp_bias: bool = False
    rms_norm_eps: float = 1e-5
    torch_dtype: str = "float16"            # "float16" | "bfloat16" | …
    pretraining_tp: int = 1

    # ─────────────────────────── Token IDs ─────────────────────────── #
    bos_token_id: int = 1
    eos_token_id: Union[int, List[int]] = 2
    pad_token_id: Optional[int] = None

    # ─────────────────────────── 雜項 ─────────────────────────── #
    tie_weights: bool = True
    architectures: Optional[List[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    devices: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # JSON helpers                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def _map_hf_keys(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        """將 Hugging Face 命名轉成本專案欄位，並回傳新 dict。"""
        key_map = {
            "num_hidden_layers": "num_layers",
            "num_attention_heads": "num_heads",
            "num_key_value_heads": "num_kv_heads",
            "hidden_size": "hidden_size",
            "intermediate_size": "intermediate_size",
            "vocab_size": "vocab_size",
            "max_position_embeddings": "max_position_embeddings",
            "head_dim": "head_dim",
            "rope_theta": "rope_theta",
        }
        remap = {key_map.get(k, k): v for k, v in raw.items()}
        return remap

    @classmethod
    def from_json(cls, path: str | Path) -> "LlamaConfig":
        """Load configuration from a JSON file, supporting HF 格式."""
        with open(path, "r", encoding="utf-8") as file:
            raw_data: Dict[str, Any] = json.load(file)

        data = cls._map_hf_keys(raw_data)
        # 自動補上 head_dim
        if "head_dim" not in data or data["head_dim"] is None:
            data["head_dim"] = data["hidden_size"] // data["num_heads"]

        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        """Dump configuration to JSON (project-native欄位)."""
        Path(path).write_text(
            json.dumps(self.__dict__, indent=2, ensure_ascii=False)
        )
