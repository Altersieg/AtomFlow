"""
AtomFlow Weight Exporter — Llama 3.2 3B
========================================
Pipeline:
  1. Offline AWQ calibration  — capture per-channel activation scales via hooks
  2. AWQ smoothing in-place   — balance act/weight outliers (LayerNorm ÷ S, W × S)
  3. Fused QKV concat         — cat(q,k,v, dim=0) before any quantization
  4. Group-wise FP8 quant     — E4M3, GS=128, scales stored as FP16
  5. Binary export            — little-endian header + 256-byte aligned tensor blocks

Output consumed by AtomFlow C++ engine via mmap.
"""

import struct
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

# ============================================================================
# Global constants / 全局常量
# ============================================================================
GROUP_SIZE = 128    # FP8 quantization group size along the K (input) dimension
                    # FP8 量化沿 K（输入）维度的分组大小
FP8_MAX    = 448.0  # Max representable value in E4M3 format / E4M3 格式的最大表示值
FILE_ALIGN = 256    # Byte alignment boundary for mmap safety / mmap 对齐边界（字节）
AWQ_ALPHA  = 0.5    # Smoothing exponent: S = act^alpha / weight^(1-alpha)
                    # 平滑指数：S = act^alpha / weight^(1-alpha)

# [Bug/Imperfection: Using 8 hardcoded sentences instead of a real WikiText-2
#  calibration set (recommended ≥ 128 samples). This under-estimates activation
#  outlier magnitude, producing slightly suboptimal smooth factors.
#  To fix: replace with `datasets.load_dataset("wikitext","wikitext-2-raw-v1")`.
#  使用了8句硬编码句子代替真实 WikiText-2 校准集（推荐≥128样本）。
#  这会低估激活异常值量级，导致平滑因子略次优。
#  修复方法：替换为 datasets.load_dataset("wikitext","wikitext-2-raw-v1")]
CALIB_CORPUS: List[str] = [
    "The Eiffel Tower is located in Paris and was built in 1889.",
    "Machine learning models require large amounts of data to train effectively.",
    "CUDA enables parallel computation on NVIDIA graphics processing units.",
    "The transformer architecture relies on self-attention mechanisms.",
    "Large language models have demonstrated impressive zero-shot capabilities.",
    "Memory bandwidth is often the bottleneck in modern deep learning inference.",
    "Quantization reduces model size by representing weights with fewer bits.",
    "The attention mechanism allows models to focus on relevant input tokens.",
]


# ============================================================================
# SECTION 1 — AWQ Offline Calibration
# 第一节 — AWQ 离线校准
# ============================================================================

def collect_activation_scales(
    model,
    tokenizer,
    device: torch.device,
    n_samples: int = 8,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Register forward-pre hooks on q_proj and gate_proj to capture the
    per-channel activation maxima flowing into each linear projection.

    Returns a dict: { layer_idx: {'attn_in': Tensor[H], 'mlp_in': Tensor[H]} }
    where H = hidden_size.

    在 q_proj 和 gate_proj 上注册前向预钩，捕获流入每个线性投影的
    逐通道激活最大值。
    返回字典：{ 层索引: {'attn_in': Tensor[H], 'mlp_in': Tensor[H]} }
    """
    hidden = model.config.hidden_size
    # Initialize accumulators with zeros; we will take running max.
    # 用零初始化累加器，逐步取最大值。
    act_scales: Dict[int, Dict[str, torch.Tensor]] = {
        i: {
            "attn_in": torch.zeros(hidden),
            "mlp_in":  torch.zeros(hidden),
        }
        for i in range(model.config.num_hidden_layers)
    }

    hooks = []

    for idx, layer in enumerate(model.model.layers):

        # Closure must capture idx by value — use default argument trick.
        # 闭包必须按值捕获 idx —— 使用默认参数技巧。
        def _make_attn_hook(i: int):
            def _hook(module: nn.Module, inp: tuple) -> None:
                # inp[0]: [batch, seq_len, hidden] — activation entering q_proj
                #         进入 q_proj 的激活张量
                with torch.no_grad():
                    x = inp[0].detach().float()
                    cur_max = x.view(-1, x.shape[-1]).abs().max(dim=0).values.cpu()
                    act_scales[i]["attn_in"] = torch.maximum(
                        act_scales[i]["attn_in"], cur_max
                    )
            return _hook

        def _make_mlp_hook(i: int):
            def _hook(module: nn.Module, inp: tuple) -> None:
                # inp[0]: [batch, seq_len, hidden] — activation entering gate_proj
                #         进入 gate_proj 的激活张量
                with torch.no_grad():
                    x = inp[0].detach().float()
                    cur_max = x.view(-1, x.shape[-1]).abs().max(dim=0).values.cpu()
                    act_scales[i]["mlp_in"] = torch.maximum(
                        act_scales[i]["mlp_in"], cur_max
                    )
            return _hook

        hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(_make_attn_hook(idx)))
        hooks.append(layer.mlp.gate_proj.register_forward_pre_hook(_make_mlp_hook(idx)))

    # --- Run calibration forward passes ---
    # --- 执行校准前向传播 ---
    model.eval()
    with torch.no_grad():
        for text in tqdm(CALIB_CORPUS[:n_samples], desc="Calibration / 校准"):
            tokens = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            model(**tokens)

    for h in hooks:
        h.remove()

    return act_scales


# ============================================================================
# SECTION 2 — AWQ Smoothing (in-place)
# 第二节 — AWQ 平滑（原地修改）
# ============================================================================

def apply_awq_smoothing(
    model,
    act_scales: Dict[int, Dict[str, torch.Tensor]],
    alpha: float = AWQ_ALPHA,
) -> None:
    """
    Apply the AWQ equivalent transformation in-place:
      S = clamp(act_scales^alpha / weight_scales^(1-alpha), min=1e-5)
      W_smooth[:, i] = W[:, i] * S[i]    (each input channel scaled up)
      gamma_smooth[i] = gamma[i] / S[i]  (layernorm compensates, math-equivalent)

    Targets per layer:
      - Attention:  input_layernorm, fused (q + k + v) weights
      - MLP:        post_attention_layernorm, gate_proj + up_proj weights

    [Bug/Imperfection: o_proj and down_proj are NOT smoothed.
     o_proj input = attention output (after softmax+V), requires dedicated output
     activation hooks. down_proj input = SwiGLU output; smoothing would distort
     the nonlinearity. Both are left as AbsMax-only quantization targets.
     o_proj 和 down_proj 未平滑。o_proj 的输入是注意力输出（softmax+V后），
     需要专门的输出激活钩子。down_proj 的输入是 SwiGLU 输出，平滑会扭曲非线性。
     两者均保留为纯 AbsMax 量化目标。]

    原地应用 AWQ 等价变换：
      S = clamp(act_scales^alpha / weight_scales^(1-alpha), min=1e-5)
      W_smooth[:, i] = W[:, i] * S[i]    （每个输入通道按 S 放大）
      gamma_smooth[i] = gamma[i] / S[i]  （LayerNorm 补偿，数学等价）
    """
    for layer_idx, layer in enumerate(
        tqdm(model.model.layers, desc="AWQ Smoothing / 平滑")
    ):
        sc = act_scales[layer_idx]
        dev = layer.self_attn.q_proj.weight.device

        # ---- Attention block ----
        q_w = layer.self_attn.q_proj.weight.detach().float().cpu()  # [q_out, H]
        k_w = layer.self_attn.k_proj.weight.detach().float().cpu()  # [k_out, H]
        v_w = layer.self_attn.v_proj.weight.detach().float().cpu()  # [v_out, H]
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)                  # [qkv_out, H]

        # Per-channel weight scale: max over output dim → shape [H]
        # 每通道权重尺度：沿输出维度取最大 → 形状 [H]
        w_scale_attn = qkv_w.abs().max(dim=0).values.clamp(min=1e-5)
        a_scale_attn = sc["attn_in"].clamp(min=1e-5)

        S_attn = (
            a_scale_attn.pow(alpha) / w_scale_attn.pow(1.0 - alpha)
        ).clamp(min=1e-5).to(dev)

        # Scale each column of q/k/v weights by S
        # 用 S 缩放 q/k/v 权重的每一列
        layer.self_attn.q_proj.weight.data.mul_(S_attn.unsqueeze(0))
        layer.self_attn.k_proj.weight.data.mul_(S_attn.unsqueeze(0))
        layer.self_attn.v_proj.weight.data.mul_(S_attn.unsqueeze(0))

        # Divide LayerNorm gamma by S — mathematically equivalent transformation
        # 将 LayerNorm gamma 除以 S —— 数学等价变换
        layer.input_layernorm.weight.data.div_(S_attn)

        # ---- MLP block ----
        gate_w = layer.mlp.gate_proj.weight.detach().float().cpu()  # [ffn_out, H]
        up_w   = layer.mlp.up_proj.weight.detach().float().cpu()    # [ffn_out, H]
        gu_w   = torch.cat([gate_w, up_w], dim=0)                   # [2*ffn_out, H]

        w_scale_mlp = gu_w.abs().max(dim=0).values.clamp(min=1e-5)
        a_scale_mlp = sc["mlp_in"].clamp(min=1e-5)

        S_mlp = (
            a_scale_mlp.pow(alpha) / w_scale_mlp.pow(1.0 - alpha)
        ).clamp(min=1e-5).to(dev)

        layer.mlp.gate_proj.weight.data.mul_(S_mlp.unsqueeze(0))
        layer.mlp.up_proj.weight.data.mul_(S_mlp.unsqueeze(0))
        layer.post_attention_layernorm.weight.data.div_(S_mlp)


# ============================================================================
# SECTION 3 — FP8 Group-wise Quantization
# 第三节 — FP8 分组量化
# ============================================================================

def quantize_fp8_gs128(tensor: torch.Tensor) -> Tuple[bytes, bytes]:
    """
    Quantize a 2-D weight tensor [N, K] to FP8 E4M3 with GS=128.
    K must be divisible by GROUP_SIZE.

    Returns:
      fp8_bytes   — raw uint8 blob, shape [N, K] packed row-major
      scales_bytes — FP16 scales blob, shape [N, K/128] packed row-major

    [Bug/Imperfection: AbsMax is used for groups not covered by AWQ smoothing
     (o_proj, down_proj). A more accurate approach would be GPTQ per-block
     Hessian-weighted quantization, but that requires calibration data per weight.
     对于 AWQ 平滑未覆盖的组（o_proj, down_proj），使用了 AbsMax 量化。
     更精确的方法是 GPTQ 每块 Hessian 加权量化，但这需要每个权重的校准数据。]

    将 2D 权重矩阵 [N, K] 量化为 FP8 E4M3，分组大小 GS=128。
    K 必须能被 GROUP_SIZE 整除。
    """
    t = tensor.detach().float()
    N, K = t.shape
    assert K % GROUP_SIZE == 0, (
        f"K={K} is not divisible by GROUP_SIZE={GROUP_SIZE} / "
        f"K={K} 无法被 GROUP_SIZE={GROUP_SIZE} 整除"
    )

    num_groups = K // GROUP_SIZE
    t_grouped  = t.view(N, num_groups, GROUP_SIZE)  # [N, G, GS]

    # AbsMax scale per group / 每组 AbsMax 尺度
    max_vals = t_grouped.abs().max(dim=2).values     # [N, G]
    scales   = torch.where(
        max_vals > 0,
        max_vals / FP8_MAX,
        torch.ones_like(max_vals),
    )

    # Quantize to E4M3 — clamp first to prevent NaN on boundary values
    # 量化到 E4M3 —— 先 clamp 防止边界值产生 NaN
    quantized = (
        (t_grouped / scales.unsqueeze(2)).clamp(-448.0, 448.0)
    ).to(torch.float8_e4m3fn)

    fp8_bytes = quantized.contiguous().view(torch.uint8).cpu().numpy().tobytes()

    # Save scales as FP16 (not FP32) — halves dequant bandwidth overhead in C++
    # 保存 Scale 为 FP16（非 FP32）—— 将 C++ 反量化带宽开销减半
    scales_bytes = scales.to(torch.float16).cpu().numpy().tobytes()

    return fp8_bytes, scales_bytes


# ============================================================================
# SECTION 4 — File I/O Helpers
# 第四节 — 文件 I/O 辅助
# ============================================================================

def _write_padded(f, data: bytes, align: int = FILE_ALIGN) -> None:
    """
    Write `data` to file `f`, then insert zero-padding until f.tell()
    is a multiple of `align`. Ensures 256-byte alignment for mmap reads.

    写入 data 后，用零填充直到 f.tell() 是 align 的倍数。
    确保 mmap 读取时的 256 字节对齐。
    """
    f.write(data)
    remainder = f.tell() % align
    if remainder:
        f.write(b"\x00" * (align - remainder))


def _write_tensor_fp32(f, tensor: torch.Tensor) -> None:
    """Write a tensor as raw FP32 with 256-byte alignment.
    以 FP32 格式写入张量，附 256 字节对齐填充。"""
    _write_padded(f, tensor.detach().float().cpu().numpy().tobytes())


def _write_tensor_fp8(f, tensor: torch.Tensor) -> None:
    """
    Quantize tensor to FP8 GS=128, then write:
      [FP8 data | 256-byte padding | FP16 scales | 256-byte padding]

    将张量量化为 FP8 GS=128，然后按以下顺序写入：
      [FP8 数据 | 256 字节填充 | FP16 Scale | 256 字节填充]
    """
    fp8_bytes, scales_bytes = quantize_fp8_gs128(tensor)
    _write_padded(f, fp8_bytes)     # align after FP8 blob  / FP8 数据后对齐
    _write_padded(f, scales_bytes)  # align after scales blob / Scale 数据后对齐


# ============================================================================
# SECTION 5 — Main Export
# 第五节 — 主导出逻辑
# ============================================================================

def export_atomflow(hf_model, filepath: str) -> None:
    """
    Export a post-AWQ-smoothed Llama 3.2 model to AtomFlow binary format.

    File layout:
      [256-byte header]
      [embed_tokens — FP32, padded]
      For each layer:
        [input_layernorm — FP32, padded]      (post-smoothing)
        [post_attention_layernorm — FP32, padded]  (post-smoothing)
        [fused_qkv_weight — FP8 + FP16 scales]   ← NEW: q/k/v concatenated
        [o_proj_weight    — FP8 + FP16 scales]
        [gate_proj_weight — FP8 + FP16 scales]
        [up_proj_weight   — FP8 + FP16 scales]
        [down_proj_weight — FP8 + FP16 scales]
      [model.norm — FP32, padded]
      [lm_head    — FP32, padded]

    将经过 AWQ 平滑的 Llama 3.2 模型导出为 AtomFlow 二进制格式。
    """
    sd     = hf_model.state_dict()
    cfg    = hf_model.config

    dim         = cfg.hidden_size
    ffn_dim     = cfg.intermediate_size
    n_layers    = cfg.num_hidden_layers
    n_heads     = cfg.num_attention_heads
    n_kv_heads  = getattr(cfg, "num_key_value_heads", n_heads)
    vocab_size  = cfg.vocab_size
    max_seq_len = getattr(cfg, "max_position_embeddings", 2048)

    with open(filepath, "wb") as f:

        # ------------------------------------------------------------------ #
        # 1. Header (256 bytes, little-endian) / 文件头（256 字节，小端序）   #
        # ------------------------------------------------------------------ #
        # Magic "ATOM" + version + group_size + 7 model params
        # 魔数 "ATOM" + 版本号 + 分组大小 + 7 个模型参数
        f.write(struct.pack("<4s", b"ATOM"))
        f.write(struct.pack("<i", 1))           # version / 版本
        f.write(struct.pack("<i", GROUP_SIZE))
        f.write(struct.pack(
            "<iiiiiii",
            dim, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len,
        ))
        pad = 256 - f.tell()
        f.write(b"\x00" * pad)

        # ------------------------------------------------------------------ #
        # 2. Token Embedding (FP32 — kept full precision for semantic fidelity)
        # Token 嵌入（FP32 —— 保留全精度以保证语义保真度）
        # ------------------------------------------------------------------ #
        print("Writing embedding / 写入嵌入层...")
        _write_tensor_fp32(f, sd["model.embed_tokens.weight"])

        # ------------------------------------------------------------------ #
        # 3. Transformer layers / Transformer 层
        # ------------------------------------------------------------------ #
        for i in tqdm(range(n_layers), desc="Exporting layers / 导出网络层"):
            p = f"model.layers.{i}."

            # --- Norm weights (FP32, post-smoothing) ---
            # --- Norm 权重（FP32，经 AWQ 平滑后）---
            _write_tensor_fp32(f, sd[p + "input_layernorm.weight"])
            _write_tensor_fp32(f, sd[p + "post_attention_layernorm.weight"])

            # --- Fused QKV weight ---
            # Concat q/k/v along output dim BEFORE quantization.
            # C++ launch_linear_gemm expects [QKV_Total_Dim, Hidden_Dim].
            # 在量化之前沿输出维度拼接 q/k/v。
            # C++ launch_linear_gemm 期望输入形状 [QKV_Total_Dim, Hidden_Dim]。
            fused_qkv = torch.cat([
                sd[p + "self_attn.q_proj.weight"],  # [q_out, H]
                sd[p + "self_attn.k_proj.weight"],  # [k_out, H]
                sd[p + "self_attn.v_proj.weight"],  # [v_out, H]
            ], dim=0)                               # [qkv_out, H]
            _write_tensor_fp8(f, fused_qkv)

            # --- o_proj, gate, up, down ---
            # [Bug/Imperfection: o_proj and down_proj are quantized with AbsMax
            #  only (no AWQ smoothing), because their input activations are not
            #  easily captured without additional output hooks.
            #  o_proj 和 down_proj 仅用 AbsMax 量化（无 AWQ 平滑），
            #  因为其输入激活无法在不添加额外输出钩子的情况下轻易捕获。]
            _write_tensor_fp8(f, sd[p + "self_attn.o_proj.weight"])
            _write_tensor_fp8(f, sd[p + "mlp.gate_proj.weight"])
            _write_tensor_fp8(f, sd[p + "mlp.up_proj.weight"])
            _write_tensor_fp8(f, sd[p + "mlp.down_proj.weight"])

        # ------------------------------------------------------------------ #
        # 4. Final norm + LM head / 最终 Norm 层 + 语言模型头
        # ------------------------------------------------------------------ #
        print("Writing output layers / 写入输出层...")
        _write_tensor_fp32(f, sd["model.norm.weight"])

        # lm_head may be weight-tied to embed_tokens
        # lm_head 可能与 embed_tokens 共享权重
        lm_src = (
            sd["lm_head.weight"]
            if "lm_head.weight" in sd
            else sd["model.embed_tokens.weight"]
        )
        _write_tensor_fp32(f, lm_src)

    print(f"Export complete / 导出完毕: {filepath}")
    print(f"File size: {Path(filepath).stat().st_size / 1024**3:.2f} GiB")


# ============================================================================
# SECTION 6 — Entry Point
# 第六节 — 入口
# ============================================================================

def main() -> None:
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Export Llama 3.2 to AtomFlow FP8 format")
    parser.add_argument("--model",   required=True, help="HuggingFace model id or local path")
    parser.add_argument("--output",  default="llama3_2_atomflow.bin", help="Output .bin file")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=8, help="Calibration sample count")
    parser.add_argument("--alpha",   type=float, default=AWQ_ALPHA, help="AWQ smoothing exponent")
    parser.add_argument("--skip-awq", action="store_true", help="Skip AWQ, use raw AbsMax only")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading model from: {args.model}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()

    if not args.skip_awq:
        print("\n[Step 1/3] Collecting activation scales / 采集激活尺度...")
        act_scales = collect_activation_scales(model, tokenizer, device, args.samples)

        print("\n[Step 2/3] Applying AWQ smoothing / 应用 AWQ 平滑...")
        apply_awq_smoothing(model, act_scales, alpha=args.alpha)
    else:
        print("[AWQ skipped] Using raw AbsMax quantization / 跳过 AWQ，使用原始 AbsMax 量化")

    print("\n[Step 3/3] Exporting weights / 导出权重...")
    export_atomflow(model, args.output)


if __name__ == "__main__":
    main()
