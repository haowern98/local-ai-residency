#!/usr/bin/env python3
"""Tokenizes a text fixture for an advanced ONNX LLM residency plan.

The normal ONNX LLM path is prompt_file=... plus tokenizer=..., which lets
MosaicVRAM tokenize internally. This helper remains useful for diagnostics that
need a fixed token file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer path")
    parser.add_argument("--input", required=True, help="Text prompt file")
    parser.add_argument("--output", required=True, help="Token ID output file")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow AutoTokenizer to download missing tokenizer files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = Path(args.input).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=not args.allow_download
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        ",".join(str(token_id) for token_id in token_ids), encoding="ascii"
    )
    print(f"tokenizer={args.tokenizer}")
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"token_count={len(token_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
