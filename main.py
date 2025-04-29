#!/usr/bin/env python
"""Command-line interface for text generation with Llama-Triton.

Example
-------
$ ./main.py /path/to/model_dir \
    --prompt "Once upon a time" --max_tokens 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

from api.inference import LlamaGenerator
from rich import print as rprint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Llama-Triton text generator")
    parser.add_argument("model_dir", type=str, help="Directory containing the model files")
    parser.add_argument(
        "-p",
        "--prompt",
        default="Hello",
        help="Prompt to prime the model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the CLI."""
    args = parse_args()

    model_dir = Path(args.model_dir)
    generator = LlamaGenerator(model_dir)

    output = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
    )

    rprint(f"[bold green]› {output}")


if __name__ == "__main__":
    main()
