"""Llama text-generation utility built on Triton-accelerated inference.

The generator

1. Loads a model configuration (`model.json`) and tokenizer.
2. Restores weights from `state_dict.pt` if available.
3. Performs autoregressive decoding with temperature + nucleus sampling.

"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch

from graph.config import LlamaConfig
from graph.model import LlamaTritonModel
from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


class LlamaGenerator:
    """High-level wrapper for prompt-based text generation."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device = "cuda",
    ) -> None:
        model_dir = Path(model_dir)

        self.config: LlamaConfig = LlamaConfig.from_json(model_dir / "model.json")
        self.tokenizer = SentencePieceTokenizer(model_dir / "tokenizer.model")

        self.model = LlamaTritonModel(self.config, device=device)

        state_dict_path = model_dir / "state_dict.pt"
        if state_dict_path.exists():
            self.model.load_state_dict(
                torch.load(state_dict_path, map_location=device),
                strict=False,
            )

        self.model.eval()  # Switch to inference mode.

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        """Generate text continuations for a given prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Softmax temperature (> 0).  Higher = more randomness.
            top_p: Nucleus-sampling threshold (0 < top_p ≤ 1).

        Returns:
            The generated text (without the prompt).
        """
        # Encode prompt and prime the model / KV cache.
        prompt_ids: List[int] = self.tokenizer.encode(
            prompt,
            add_bos=True,
            add_eos=False,
        )
        input_ids = torch.tensor(prompt_ids, device=self.model.device)
        _ = self.model(input_ids)  # Warm-up forward pass.

        current_token: int = prompt_ids[-1]
        generated_ids: List[int] = []

        for _ in range(max_new_tokens):
            logits = self.model(
                torch.tensor([current_token], device=self.model.device)
            ) / temperature

            probs = torch.softmax(logits, dim=-1)

            # Nucleus (top-p) sampling.
            sorted_idx = torch.argsort(probs, descending=True)
            cumulative = torch.cumsum(probs[sorted_idx], dim=0)
            kept_idx = sorted_idx[cumulative <= top_p]

            # Edge case: ensure at least one token is kept.
            if kept_idx.numel() == 0:
                kept_idx = sorted_idx[:1]

            current_token = kept_idx[
                torch.multinomial(probs[kept_idx], num_samples=1)
            ].item()

            # Stop if EOS token is produced.
            if current_token == self.tokenizer.eos_id:
                break

            generated_ids.append(current_token)

        return self.tokenizer.decode(generated_ids)
