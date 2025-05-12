"""api.inference

Llama-Triton 文字生成介面：
    • 支援 native Triton 權重 (state_dict.pt + model.json)
    • 支援 Hugging Face 檢查點 (需使用 fmt="hf" 明確指定)

不再自動推斷格式——由使用者決定。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch

from graph.config import LlamaConfig
from graph.model import LlamaTritonModel
from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer

# optional HF converter
try:
    from utils.hf_import import convert as hf_convert
except ModuleNotFoundError:
    hf_convert = None


class LlamaGenerator:
    """Prompt-based text generator without implicit auto-convert."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        fmt: str = "triton",          # "triton" | "hf"
        device: str | torch.device = "cuda",
        dtype: str | None = None,     # override dtype if needed
    ) -> None:
        """初始化

        Args
        ----
        model_dir :  權重資料夾。
        fmt       :  "triton" → 直接載入；"hf" → 轉換後載入。
        device    :  CUDA / CPU 等。
        dtype     :  強制覆寫 config.torch_dtype (e.g. "bfloat16")。
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.fmt = fmt.lower()

        if self.fmt not in {"triton", "hf"}:
            raise ValueError('`fmt` 必須是 "triton" 或 "hf"')

        if self.fmt == "hf":
            if hf_convert is None:
                raise ImportError("未安裝 utils.hf_import，無法處理 Hugging Face 權重。")
            hf_convert(self.model_dir, target_dtype=dtype or "float16")

        # ─────────────────── 讀 config / tokenizer ─────────────────── #
        cfg_path = self.model_dir / "model.json"
        if not cfg_path.exists():
            raise FileNotFoundError("找不到 model.json，請確認 fmt 與資料夾內容。")

        self.config: LlamaConfig = LlamaConfig.from_json(cfg_path)
        if dtype is not None:
            self.config.torch_dtype = dtype

        tok_path = self.model_dir / "tokenizer.model"
        if not tok_path.exists():
            raise FileNotFoundError("缺少 tokenizer.model (SentencePiece)")
        self.tokenizer = SentencePieceTokenizer(tok_path)

        # ─────────────────── 載入 state_dict ─────────────────── #
        sd_path = self.model_dir / "state_dict.pt"
        if not sd_path.exists():
            raise FileNotFoundError("state_dict.pt 不存在，無法載入權重。")

        self.model = LlamaTritonModel(self.config, device=self.device)
        self.model.load_state_dict(
            torch.load(sd_path, map_location=self.device), strict=False
        )
        self.model.eval()

        eos_ids = self.config.eos_token_id
        self.eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]

    # ------------------------------------------------------------------ #
    # Generation                                                         #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        """以 nucleus + temperature sampling 產生續句。"""
        prompt_ids: List[int] = self.tokenizer.encode(
            prompt, add_bos=True, add_eos=False
        )
        input_ids = torch.tensor(prompt_ids, device=self.device)
        _ = self.model(input_ids)  # prime KV-cache

        cur = prompt_ids[-1]
        out_ids: List[int] = []

        for _ in range(max_new_tokens):
            logits = self.model(torch.tensor([cur], device=self.device)) / temperature
            probs = torch.softmax(logits, dim=-1)

            sorted_idx = torch.argsort(probs, descending=True)
            cumulative = torch.cumsum(probs[sorted_idx], 0)
            kept = sorted_idx[cumulative <= top_p]
            kept = kept if kept.numel() else sorted_idx[:1]

            cur = kept[torch.multinomial(probs[kept], 1)].item()
            if cur in self.eos_ids:
                break
            out_ids.append(cur)

        return self.tokenizer.decode(out_ids)
