#!/usr/bin/env python3
"""
migrate_lmhead_fp16.py — Convert lm_head from FP32 to FP16 in an existing
AtomFlow .bin weight file.  The file is rewritten in-place (shrinks by ~0.75 GiB).

将已有 AtomFlow .bin 权重文件中的 lm_head 从 FP32 就地转换为 FP16。
文件将缩小约 0.75 GiB。

Usage / 用法:
    python tools/migrate_lmhead_fp16.py [path-to-bin]
    # default: models/llama3_2_atomflow.bin
"""

import struct
import sys
import os
import numpy as np
from pathlib import Path

FILE_ALIGN = 256

def align_up(sz: int, a: int = FILE_ALIGN) -> int:
    return (sz + a - 1) & ~(a - 1)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "models/llama3_2_atomflow.bin"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    # ── 1. Parse header / 解析文件头 ─────────────────────────────────────
    with open(path, "rb") as f:
        hdr = f.read(256)
    vals = struct.unpack_from("<4s i i iiiiiii", hdr, 0)
    magic = vals[0]
    assert magic == b"ATOM", f"Bad magic: {magic}"
    GS         = vals[2]
    dim        = vals[3]
    hidden_dim = vals[4]
    n_layers   = vals[5]
    n_heads    = vals[6]
    n_kv_heads = vals[7]
    vocab_size = vals[8]

    HD      = dim // n_heads
    Q_DIM   = dim
    KV_DIM  = n_kv_heads * HD
    QKV_OUT = Q_DIM + KV_DIM + KV_DIM

    print(f"[Header]  dim={dim} hidden_dim={hidden_dim} NL={n_layers} "
          f"NH={n_heads} NKV={n_kv_heads} V={vocab_size} GS={GS}")

    # ── 2. Compute lm_head offset / 计算 lm_head 偏移 ────────────────────
    cursor = 256
    cursor += align_up(vocab_size * dim * 4)          # embed_tokens FP32

    for _ in range(n_layers):
        cursor += align_up(dim * 4)                    # input_norm FP32
        cursor += align_up(dim * 4)                    # post_norm  FP32
        cursor += align_up(QKV_OUT * dim)              # qkv FP8
        cursor += align_up(QKV_OUT * (dim // GS) * 2)  # qkv scales FP16
        cursor += align_up(dim * dim)                  # o_proj FP8
        cursor += align_up(dim * (dim // GS) * 2)      # o_proj scales
        cursor += align_up(hidden_dim * dim)           # gate FP8
        cursor += align_up(hidden_dim * (dim // GS) * 2)  # gate scales
        cursor += align_up(hidden_dim * dim)           # up FP8
        cursor += align_up(hidden_dim * (dim // GS) * 2)  # up scales
        cursor += align_up(dim * hidden_dim)           # down FP8
        cursor += align_up(dim * (hidden_dim // GS) * 2)  # down scales

    cursor += align_up(dim * 4)                        # model.norm FP32

    lm_head_offset = cursor
    lm_head_numel  = vocab_size * dim
    lm_head_fp32_bytes = lm_head_numel * 4
    lm_head_fp16_bytes = lm_head_numel * 2

    file_size = os.path.getsize(path)
    expected_end = lm_head_offset + lm_head_fp32_bytes
    assert expected_end == file_size or expected_end == align_up(lm_head_offset + lm_head_fp32_bytes), \
        f"Offset mismatch: expected {expected_end}, file size {file_size}"

    print(f"[lm_head]  offset={lm_head_offset}  "
          f"FP32={lm_head_fp32_bytes / 1024**3:.3f} GiB  →  "
          f"FP16={lm_head_fp16_bytes / 1024**3:.3f} GiB")

    # ── 3. Read FP32, convert to FP16, write back / 读取并转换 ─────────
    # Process in chunks to avoid >1.5 GB single allocation
    # 分块处理，避免单次分配 >1.5 GB 内存
    CHUNK = 64 * 1024 * 1024  # 64M floats = 256 MB per chunk / 每块 256 MB

    with open(path, "r+b") as f:
        read_pos  = lm_head_offset
        write_pos = lm_head_offset
        remaining = lm_head_numel

        while remaining > 0:
            n = min(CHUNK, remaining)
            f.seek(read_pos)
            buf = f.read(n * 4)
            fp32 = np.frombuffer(buf, dtype=np.float32)
            fp16 = fp32.astype(np.float16)

            f.seek(write_pos)
            f.write(fp16.tobytes())

            read_pos  += n * 4
            write_pos += n * 2
            remaining -= n

        # Add alignment padding / 添加对齐填充
        remainder = write_pos % FILE_ALIGN
        if remainder:
            f.write(b"\x00" * (FILE_ALIGN - remainder))
            write_pos += FILE_ALIGN - remainder

        # Truncate file / 截断文件
        f.truncate(write_pos)

    new_size = os.path.getsize(path)
    saved = file_size - new_size
    print(f"[Done]  {path}")
    print(f"  Old size: {file_size / 1024**3:.2f} GiB")
    print(f"  New size: {new_size / 1024**3:.2f} GiB")
    print(f"  Saved:    {saved / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
