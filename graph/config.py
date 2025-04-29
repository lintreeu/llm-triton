"""Llama configuration dataclass.

Defines model-wide hyperparameters and provides JSON (de)serialization
helpers. 
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LlamaConfig:
    """Container for all model hyperparameters."""

    # ----------------------------- Model size ----------------------------- #
    num_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int

    # ----------------------------- Rotary Embedding ----------------------- #
    rope_freq_base: float = 10_000.0
    rope_freq_scale: float = 1.0

    # ----------------------------- KV cache -------------------------------- #
    page_size: int = 512
    max_pages: int = 2_048
    kv_dtype: str = "float16"  # 'float16' or 'bfloat16'

    # ----------------------------- Tokens ---------------------------------- #
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: Optional[int] = None

    # ----------------------------- Misc ------------------------------------ #
    tie_weights: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)
    devices: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------------- #
    # JSON helpers                                                           #
    # ---------------------------------------------------------------------- #

    @classmethod
    def from_json(cls, path: str | Path) -> "LlamaConfig":
        """Load a configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        """Serialize the configuration to a JSON file."""
        Path(path).write_text(json.dumps(self.__dict__, indent=2, ensure_ascii=False))
