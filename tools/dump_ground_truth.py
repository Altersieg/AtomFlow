"""
dump_ground_truth.py
Runs a fixed prompt through HuggingFace Llama 3.2 3B, captures intermediate
activations via forward hooks, and saves them as raw FP32 binary files.

运行固定 prompt 通过 HuggingFace Llama 3.2 3B，用 forward hook 捕获中间激活，
并将其保存为原始 FP32 二进制文件。

Usage / 用法:
    python tools/dump_ground_truth.py \
        --model_path /path/to/llama-3.2-3b \
        --output_dir ground_truth/

Output files / 输出文件:
    gt_input_embeddings.bin    — embed_tokens output for last token position [1, D]
    gt_layer{N}_norm_in.bin    — OUTPUT of input_layernorm of layer N (matches act.x_norm)
    gt_layer{N}_attn_out.bin   — output of self_attn (after o_proj)
    gt_layer{N}_mlp_out.bin    — output of mlp block
    gt_logits.bin              — final lm_head logits (last token position)

[Bug/Imperfection: We capture hook outputs for only the LAST token position
 (index -1 of the sequence dim). If the model uses past_key_values for prefill,
 the shapes change. We force use_cache=False to keep shapes deterministic.
 仅捕获最后 token 位置（序列维度 -1）的 hook 输出。
 使用 use_cache=False 保持形状确定性。]
"""

import argparse
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import AWQ calibration & smoothing so GT uses the same smoothed model
# that the C++ engine's exported weights came from.
# 导入 AWQ 校准和平滑，使 GT 使用与 C++ 引擎导出权重相同的平滑模型。
sys.path.insert(0, os.path.dirname(__file__))
from export_atomflow import collect_activation_scales, apply_awq_smoothing

# Layers to instrument / 要插桩的层索引
PROBE_LAYERS = [0, 13, 27]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Path to HF model dir / HF 模型目录路径")
    p.add_argument("--output_dir", default="ground_truth",
                   help="Directory for .bin files / 输出目录")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-awq", action="store_true",
                   help="Skip AWQ smoothing (use original model) / 跳过 AWQ 平滑")
    return p.parse_args()


def save_fp32(arr: torch.Tensor, path: str):
    """Flatten tensor, cast to FP32, write raw bytes.
    将张量展平，转换为 FP32，写入原始字节。"""
    data = arr.detach().float().cpu().numpy().flatten()
    data.tofile(path)
    print(f"  [SAVED] {path}  shape={list(arr.shape)}  dtype={arr.dtype}  "
          f"numel={data.size}  bytes={data.nbytes}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load model and tokenizer
    #    加载模型和分词器
    # -----------------------------------------------------------------------
    print(f"[1] Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float32,  # FP32 for ground truth / FP32 作为基准真值
        local_files_only=True,
    ).to(args.device)
    model.eval()
    print(f"    Device: {args.device}  |  Num layers: {model.config.num_hidden_layers}")

    # -----------------------------------------------------------------------
    # 1b. Apply AWQ smoothing (same transform as export_atomflow.py)
    #     使 GT 模型与 C++ 引擎使用相同的平滑权重
    # -----------------------------------------------------------------------
    if not args.skip_awq:
        print("[1b] Running AWQ calibration + smoothing (matches export_atomflow.py) ...")
        act_scales = collect_activation_scales(model, tokenizer, torch.device(args.device))
        apply_awq_smoothing(model, act_scales)
        print("     AWQ smoothing applied — GT model now matches exported weights.")
    else:
        print("[1b] AWQ smoothing SKIPPED — GT uses original model.")

    # -----------------------------------------------------------------------
    # 2. Register forward hooks
    #    注册 forward hook
    # -----------------------------------------------------------------------
    # hook_store: maps filename_stem → captured tensor
    # hook_store：文件名 stem → 捕获的张量
    hook_store: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(key: str):
        """Return a hook closure that saves the output tensor.
        返回一个保存输出张量的 hook 闭包。"""
        def hook(module, input, output):
            # output may be a tuple (e.g. attn returns (out, weights, cache))
            # output 可能是元组（如 attn 返回 (out, weights, cache)）
            t = output[0] if isinstance(output, tuple) else output
            # Capture LAST token position; detach to avoid holding computation graph.
            # 捕获最后一个 token 位置；detach 避免持有计算图。
            hook_store[key] = t[:, -1, :].detach().clone()  # shape [1, hidden_dim]
        return hook

    # embed_tokens hook: capture the embedding output for the last token position.
    # This is what AtomFlow's act.x is initialised to (after file injection).
    # embed_tokens hook：捕获最后 token 位置的嵌入输出，与 act.x 初始值对齐。
    h = model.model.embed_tokens.register_forward_hook(
        make_hook("input_embeddings"))
    handles.append(h)

    layers = model.model.layers

    for n in PROBE_LAYERS:
        # Capture the OUTPUT of input_layernorm (post-norm), which matches
        # act.x_norm in AtomFlow C++ after launch_rms_norm().
        # 捕获 input_layernorm 的输出（post-norm），与 C++ act.x_norm 语义对齐。
        h = layers[n].input_layernorm.register_forward_hook(
            make_hook(f"layer{n}_norm_in"))
        handles.append(h)

        # self_attn output (after o_proj, before residual)
        # self_attn 输出（o_proj 之后，残差相加之前）
        h = layers[n].self_attn.register_forward_hook(make_hook(f"layer{n}_attn_out"))
        handles.append(h)

        # mlp output (before residual)
        # mlp 输出（残差相加之前）
        h = layers[n].mlp.register_forward_hook(make_hook(f"layer{n}_mlp_out"))
        handles.append(h)

    # -----------------------------------------------------------------------
    # 3. Tokenize and run forward pass
    #    分词并运行前向传播
    # -----------------------------------------------------------------------
    # Use a single BOS token (seq_len=1) so that attention has seq_kv=1,
    # matching the C++ engine's fast-path (no KV cache).
    # 使用单个 BOS token（seq_len=1），使注意力 seq_kv=1，与 C++ 引擎快速路径匹配。
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = 128000  # Llama 3 default BOS
    input_ids = torch.tensor([[bos_id]], device=args.device)
    print(f"[2] Single-token input: BOS id={bos_id}  (seq_len=1)")
    print(f"    Token IDs: {input_ids[0].tolist()}")

    with torch.no_grad():
        # [Bug/Imperfection: use_cache=False disables KV caching, making this a
        #  full prefill pass. AtomFlow's current MVP also has no KV cache, so the
        #  shapes match. When KV cache is added to AtomFlow, set use_cache=True here
        #  and capture the KV tensors separately.
        #  use_cache=False 禁用 KV 缓存，执行完整 prefill。
        #  AtomFlow 当前 MVP 同样没有 KV 缓存，形状一致。
        #  AtomFlow 添加 KV 缓存后，此处应改为 use_cache=True 并单独捕获 KV 张量。]
        outputs = model(input_ids=input_ids, use_cache=False)

    logits = outputs.logits  # shape [1, seq_len, vocab_size]

    # -----------------------------------------------------------------------
    # 4. Remove hooks, save tensors
    #    移除 hook，保存张量
    # -----------------------------------------------------------------------
    for h in handles:
        h.remove()

    print(f"[3] Saving ground truth tensors to {args.output_dir}/ ...")
    for key, tensor in hook_store.items():
        save_fp32(tensor, os.path.join(args.output_dir, f"gt_{key}.bin"))

    # Save last-token logits [1, vocab_size]
    # 保存最后 token 的 logits [1, vocab_size]
    save_fp32(logits[:, -1, :], os.path.join(args.output_dir, "gt_logits.bin"))

    # Print per-file summary / 打印每个文件摘要
    total = len(hook_store) + 1  # +1 for logits
    print(f"\n[DONE] Wrote {total} ground truth files to {args.output_dir}/")

    # -----------------------------------------------------------------------
    # 5. Print predicted next token for sanity check
    #    打印预测的下一个 token 用于健全性检查
    # -----------------------------------------------------------------------
    next_token_id = logits[0, -1, :].argmax().item()
    next_token_str = tokenizer.decode([next_token_id])
    print(f"\n[4] HF predicted next token: id={next_token_id}  str='{next_token_str}'")
    print("    (AtomFlow should match this token id for logit cosine sim > 0.99)")



if __name__ == "__main__":
    main()
