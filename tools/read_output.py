#!/usr/bin/env python3
"""
read_output.py — Offline token decoder for AtomFlow engine output
read_output.py — AtomFlow 引擎输出的离线 token 解码器

[EN] Reads integer token IDs from output_tokens.txt (one per line),
     loads the HuggingFace tokenizer for the target model, and prints
     the decoded human-readable text.

     Usage:
       python tools/read_output.py                          # default file
       python tools/read_output.py --file my_tokens.txt     # custom file
       python tools/read_output.py --model meta-llama/Llama-3.2-3B

[CN] 从 output_tokens.txt 中读取整数 token ID（每行一个），
     加载目标模型的 HuggingFace tokenizer，并打印解码后的可读文本。

     用法：
       python tools/read_output.py                          # 默认文件
       python tools/read_output.py --file my_tokens.txt     # 自定义文件
       python tools/read_output.py --model meta-llama/Llama-3.2-3B
"""

import argparse
import sys
from pathlib import Path


def main():
    # ── Argument parsing / 参数解析 ──────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Decode AtomFlow token IDs to text / 将 AtomFlow token ID 解码为文本"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default="output_tokens.txt",
        help="[EN] Path to the token ID file (one int per line). "
             "[CN] token ID 文件路径（每行一个整数）。"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/Llama-3.2-3B-Instruct",
        help="[EN] Local path to HuggingFace tokenizer directory. "
             "[CN] HuggingFace tokenizer 目录的本地路径。"
    )
    args = parser.parse_args()

    # ── Read token IDs from file / 从文件读取 token ID ───────────────────
    token_path = Path(args.file)
    if not token_path.exists():
        print(f"[ERROR] Token file not found: {token_path}", file=sys.stderr)
        print(f"[提示]  请先运行 ./build/atomflow 生成 {token_path}", file=sys.stderr)
        sys.exit(1)

    token_ids = []
    with open(token_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                token_ids.append(int(stripped))
            except ValueError:
                print(
                    f"[WARN] Skipping non-integer on line {line_num}: '{stripped}'",
                    file=sys.stderr,
                )

    if not token_ids:
        print("[ERROR] No valid token IDs found in file.", file=sys.stderr)
        sys.exit(1)

    print(f"[decode]  Read {len(token_ids)} token IDs from {token_path}")
    print(f"[decode]  IDs: {token_ids}")

    # ── Load tokenizer / 加载 tokenizer ──────────────────────────────────
    # [EN] Import here so that the file-reading logic above works even
    #      without `transformers` installed (for quick debugging).
    # [CN] 在此处导入，这样即使未安装 transformers，
    #      上面的文件读取逻辑仍然可以工作（便于快速调试）。
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "[ERROR] `transformers` package not found.\n"
            "        pip install transformers",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[decode]  Loading tokenizer: {args.model} (local only, no downloads)")
    # [EN] local_files_only=True: strictly offline, never contact HuggingFace Hub.
    # [CN] local_files_only=True: 严格离线，永不访问 HuggingFace Hub。
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    # ── Decode and print / 解码并打印 ────────────────────────────────────
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Decoded Output / 解码输出                                ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"  {decoded_text}")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
