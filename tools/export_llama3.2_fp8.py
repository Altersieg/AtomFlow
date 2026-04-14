"""
FP8 (E4M3) Group-wise (GS=128)
"""
import struct
import torch
from pathlib import Path
from tqdm import tqdm

# 全局配置
GROUP_SIZE = 128
FP8_MAX = 448.0

def quantize_fp8_gs128(tensor: torch.Tensor):
    """
    核心量化逻辑：按 GS=128 分组，计算 Scale,返回 FP8 字节流和 Scale 字节流
    传入的 tensor shape 必须是 [N, K]，且 K 能被 128 整除。
    """
    t = tensor.detach().float()
    N, K = t.shape
    assert K % GROUP_SIZE == 0, f"维度 {K} 无法被 GROUP_SIZE={GROUP_SIZE} 整除"
    
    num_groups = K // GROUP_SIZE
    
    # 物理切块: [N, num_groups, GROUP_SIZE]
    t_grouped = t.view(N, num_groups, GROUP_SIZE)
    
    # 计算每个 Group 的绝对最大值，并求出 Scale
    max_vals = t_grouped.abs().max(dim=2).values  # [N, num_groups]
    scales = torch.where(max_vals > 0, max_vals / FP8_MAX, torch.ones_like(max_vals))
    
    # 广播 Scale 并量化到 E4M3
    quantized = (t_grouped / scales.unsqueeze(2)).to(torch.float8_e4m3fn)
    
    # 转为二进制字节流 (FP8 数据 + FP32 Scales)
    fp8_bytes = quantized.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    scales_bytes = scales.cpu().numpy().tobytes()
    
    return fp8_bytes, scales_bytes

def export_atomflow(hf_model, filepath):
    # 获取参数字典和配置
    state_dict = hf_model.state_dict()
    config = hf_model.config
    
    dim = config.hidden_size
    hidden_dim = config.intermediate_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    vocab_size = config.vocab_size
    max_seq_len = getattr(config, "max_position_embeddings", 2048)

    with open(filepath, 'wb') as f:
        # ==========================================
        # 1. 写入魔数与头部 (C++ 结构体完美对齐)
        # 结构: Magic(4s), Version(i), GS(i), Dim(i), HiddenDim(i), Layers(i), Heads(i), KVHeads(i), Vocab(i), SeqLen(i)
        # ==========================================
        f.write(struct.pack('4s', b'ATOM'))  # 魔数 ATOM
        f.write(struct.pack('i', 1))         # 版本 v1
        f.write(struct.pack('i', GROUP_SIZE))
        
        header_params = struct.pack('iiiiiii', dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len)
        f.write(header_params)
        
        # 补齐到 256 字节 (给未来预留字段)
        pad_size = 256 - f.tell()
        f.write(b'\0' * pad_size)

        # ==========================================
        # 2. 数据驱动写入 (抛弃长篇幅硬编码)
        # ==========================================
        print("开始导出 AtomFlow 格式...")
        
        # Token Embedding (保持 FP32 保证语义精度)
        print("写入 Embedding...")
        f.write(state_dict["model.embed_tokens.weight"].detach().float().numpy().tobytes())
        
        # 核心 Transformer 层循环
        target_fp8_weights = [
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.down_proj.weight", "mlp.up_proj.weight"
        ]
        
        for i in tqdm(range(n_layers), desc="处理网络层"):
            prefix = f"model.layers.{i}."
            
            # 写入 Norm (FP32，极其重要，不能量化)
            f.write(state_dict[prefix + "input_layernorm.weight"].detach().float().numpy().tobytes())
            f.write(state_dict[prefix + "post_attention_layernorm.weight"].detach().float().numpy().tobytes())
            
            # 写入核心线性层 (FP8 + Scales)
            for w_name in target_fp8_weights:
                tensor = state_dict[prefix + w_name]
                fp8_data, scales_data = quantize_fp8_gs128(tensor)
                f.write(fp8_data)
                f.write(scales_data)
                
        # 最后的 Norm 和 Output
        print("写入输出层...")
        f.write(state_dict["model.norm.weight"].detach().float().numpy().tobytes())
        
        # 处理可能共享的 lm_head
        if "lm_head.weight" in state_dict:
            f.write(state_dict["lm_head.weight"].detach().float().numpy().tobytes())
        else:
            f.write(state_dict["model.embed_tokens.weight"].detach().float().numpy().tobytes())
            
    print(f"导出完毕！文件已保存至: {filepath}")

# 使用示例 (假设你已经在另一个脚本做完了 AWQ 平滑并得到了 hf_model)
# export_atomflow(hf_model, "llama3_2_atomflow.bin")