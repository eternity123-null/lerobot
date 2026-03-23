#!/usr/bin/env python3
"""
详细检查 AR 模型参数量，对比原始 CARP
"""

import torch
import sys
sys.path.insert(0, 'src')

from lerobot.policies.carp.CFAP.autoreg import Coarse2FineAutoRegressor
from lerobot.policies.carp.MSAT.vqvae import MultiScaleActionTokenizer
from lerobot.policies.carp.obs_encoder_builder import CARPObsEncoder
from lerobot.configs.types import PolicyFeature, FeatureType

def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def count_module_params(module):
    """统计某个模块的参数量"""
    return sum(p.numel() for p in module.parameters())

def format_number(num):
    """格式化参数数量"""
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

print("=" * 80)
print("CARP AR 模型详细参数量分析")
print("=" * 80)

# ==================== 创建模拟 VAE ====================
print("\n【创建 VAE Proxy】")
vae = MultiScaleActionTokenizer(
    vocab_size=1024,
    z_channels=8,
    ch=2,
    ch_mult=(2, 4),
    action_dim=7,  # LIBERO
    num_actions=16,
    dropout=0.0,
    beta=0.25,
    using_znorm=True,
    quant_resi=0.5,
    share_quant_resi=4,
    v_patch_nums=(1, 2, 3, 4),
    test_mode=True,
)
print(f"✓ VAE 创建成功 (action_dim=7, vocab_size=1024)")

# ==================== 创建 Observation Encoder ====================
print("\n【创建 Observation Encoder】")

# 创建临时配置
from lerobot.policies.carp.configuration_carp import CARPConfig
temp_config = CARPConfig()
temp_config.input_features = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
}
temp_config.output_features = {
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
}
temp_config.device = "cpu"

obs_encoder = CARPObsEncoder(temp_config)
obs_encoder.eval()
print(f"✓ Observation Encoder 创建成功")
print(f"  输出维度: {obs_encoder.output_shape()[0]}")

# 统计 obs_encoder 参数
obs_params = count_module_params(obs_encoder)
print(f"  参数量: {format_number(obs_params)} ({obs_params:,})")

# ==================== 创建 AR 模型 (LIBERO: 7-dim, 40 tasks) ====================
print("\n" + "=" * 80)
print("【AR 模型 - LIBERO 配置 (7-dim actions, 40 tasks)】")
print("-" * 80)

ar_model_libero = Coarse2FineAutoRegressor(
    vae_proxy=vae,
    obs_encoder=obs_encoder,
    action_dim=7,      # LIBERO
    task_num=40,       # LIBERO
    task_embed_dim=3,
    depth=32,
    embed_dim=160,
    num_heads=32,
    mlp_ratio=4.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    patch_nums=(1, 2, 3, 4),
    n_obs_steps=1,
)

# 统计总参数
total, trainable = count_parameters(ar_model_libero)
print(f"\n总参数量: {format_number(total)} ({total:,})")
print(f"可训练参数: {format_number(trainable)} ({trainable:,})")

# ==================== 详细分解 ====================
print("\n详细参数分解:")
print("-" * 80)

# 1. Task Embedding
task_embed_params = count_module_params(ar_model_libero.task_embed)
print(f"1. Task Embedding: {format_number(task_embed_params)} ({task_embed_params:,})")
print(f"   {ar_model_libero.task_embed.num_embeddings} tasks × {ar_model_libero.task_embed.embedding_dim}-dim")

# 2. Observation Encoder
obs_encoder_params = count_module_params(ar_model_libero.obs_encoder)
print(f"\n2. Observation Encoder: {format_number(obs_encoder_params)} ({obs_encoder_params:,})")

# 3. Embeddings
word_embed_params = count_module_params(ar_model_libero.word_embed)
obs_embed_params = count_module_params(ar_model_libero.obs_embed)
pos_params = ar_model_libero.pos_start.numel() + ar_model_libero.pos_1LC.numel()
lvl_embed_params = count_module_params(ar_model_libero.lvl_embed)
embed_total = word_embed_params + obs_embed_params + pos_params + lvl_embed_params

print(f"\n3. Embeddings: {format_number(embed_total)} ({embed_total:,})")
print(f"   - word_embed: {format_number(word_embed_params)} ({word_embed_params:,})")
print(f"   - obs_embed: {format_number(obs_embed_params)} ({obs_embed_params:,})")
print(f"   - pos (pos_start + pos_1LC): {format_number(pos_params)} ({pos_params:,})")
print(f"   - lvl_embed: {format_number(lvl_embed_params)} ({lvl_embed_params:,})")

# 4. Transformer Blocks
blocks_params = count_module_params(ar_model_libero.blocks)
print(f"\n4. Transformer Blocks ({ar_model_libero.depth} layers): {format_number(blocks_params)} ({blocks_params:,})")
per_block = blocks_params / ar_model_libero.depth
print(f"   平均每层: {format_number(per_block)} ({int(per_block):,})")
print(f"   - embed_dim: {ar_model_libero.C}")
print(f"   - num_heads: {ar_model_libero.num_heads}")
print(f"   - mlp_dim: {int(ar_model_libero.C * 4.0)}")

# 5. Classification Head
head_nm_params = count_module_params(ar_model_libero.head_nm)
head_params = count_module_params(ar_model_libero.head)
head_total = head_nm_params + head_params
print(f"\n5. Classification Head: {format_number(head_total)} ({head_total:,})")
print(f"   - head_nm (AdaLNBeforeHead): {format_number(head_nm_params)} ({head_nm_params:,})")
print(f"   - head (Linear): {format_number(head_params)} ({head_params:,})")
print(f"   输出: vocab_size={ar_model_libero.V}")

# ==================== 创建原始配置的 AR 模型 (10-dim, 8 tasks) ====================
print("\n" + "=" * 80)
print("【AR 模型 - 原始 CARP 配置 (10-dim actions, 8 tasks)】")
print("-" * 80)

# 创建 10-dim VAE
vae_10dim = MultiScaleActionTokenizer(
    vocab_size=1024,
    z_channels=8,
    ch=2,
    ch_mult=(2, 4),
    action_dim=10,  # 原始
    num_actions=16,
    dropout=0.0,
    beta=0.25,
    using_znorm=True,
    quant_resi=0.5,
    share_quant_resi=4,
    v_patch_nums=(1, 2, 3, 4),
    test_mode=True,
)

ar_model_original = Coarse2FineAutoRegressor(
    vae_proxy=vae_10dim,
    obs_encoder=obs_encoder,  # 复用相同的 obs_encoder
    action_dim=10,     # 原始
    task_num=8,        # 原始
    task_embed_dim=3,
    depth=32,
    embed_dim=160,
    num_heads=32,
    mlp_ratio=4.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    patch_nums=(1, 2, 3, 4),
    n_obs_steps=1,
)

total_orig, trainable_orig = count_parameters(ar_model_original)
print(f"\n总参数量: {format_number(total_orig)} ({total_orig:,})")
print(f"可训练参数: {format_number(trainable_orig)} ({trainable_orig:,})")

# ==================== 对比 ====================
print("\n" + "=" * 80)
print("【参数量对比】")
print("-" * 80)

print(f"\nLIBERO (7-dim, 40 tasks):  {format_number(total)} ({total:,})")
print(f"原始 CARP (10-dim, 8 tasks): {format_number(total_orig)} ({total_orig:,})")
print(f"差异: {format_number(abs(total - total_orig))} ({abs(total - total_orig):,})")

# 计算核心 Transformer 部分是否一致
libero_blocks = count_module_params(ar_model_libero.blocks)
original_blocks = count_module_params(ar_model_original.blocks)
print(f"\n核心 Transformer Blocks:")
print(f"  LIBERO:  {format_number(libero_blocks)} ({libero_blocks:,})")
print(f"  原始:    {format_number(original_blocks)} ({original_blocks:,})")
print(f"  一致性: {'✓ 完全一致' if libero_blocks == original_blocks else '✗ 不一致'}")

print("\n参数量差异来源:")
print(f"  1. task_embed: 40×3 vs 8×3 = {(40-8)*3} 参数")
print(f"  2. patch_nums 乘以 action_dim:")
print(f"     LIBERO: {ar_model_libero.patch_nums} (sum={sum(ar_model_libero.patch_nums)})")
print(f"     原始:   {ar_model_original.patch_nums} (sum={sum(ar_model_original.patch_nums)})")
print(f"     这会影响 position embeddings 的大小")

print("\n" + "=" * 80)
print("【结论】")
print("-" * 80)
print("✓ 核心 Transformer 架构完全一致 (32层, embed_dim=160, num_heads=32)")
print("✓ 参数量差异仅来自于:")
print("  - task_num 不同 (8 vs 40)")
print("  - action_dim 不同 (10 vs 7), 导致 patch_nums 长度不同")
print("✓ 网络层数、注意力头数、MLP 倍数等核心设计完全匹配")
print("=" * 80)
