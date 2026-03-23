# CARP 超参数对比分析

## 原始 CARP (multitask) vs 当前实现

### 关键差异

| 参数 | 原始 CARP | 当前实现 | 影响 | 建议 |
|------|-----------|----------|------|------|
| **vocab_size** | 1024 | 512 | ⚠️ **关键** | 改为 1024 |
| batch_size (VAE) | 500 | 默认由用户指定 | 中等 | 使用 32-64 (受 GPU 限制) |
| batch_size (AR) | 512 | 默认由用户指定 | 中等 | 使用 16-32 (受 GPU 限制) |
| epochs (VAE) | 50 | 默认由用户指定 | 低 | 使用 50 epochs 或更多 |
| epochs (AR) | 200 | 默认由用户指定 | 低 | 使用 100-200 epochs |

### 完全一致的参数

#### VAE (MSAT) 配置
| 参数 | 值 | 说明 |
|------|-----|------|
| vae_lr | 3e-4 | ✅ 学习率 |
| vae_weight_decay | 0.005 | ✅ 权重衰减 |
| vocab_ch | 8 | ✅ Latent 通道数 |
| vch | 2 | ✅ VAE 基础通道数 |
| ch_mult | (2, 4) | ✅ 下采样倍数 |
| vqbeta | 0.25 | ✅ VQ commitment 权重 |
| vqnorm | True | ✅ 使用 cosine similarity |
| vqresi | 0.5 | ✅ Residual 权重 |
| vae_dropout | 0.0 | ✅ Dropout |
| action_horizon | 16 | ✅ 动作序列长度 |
| patch_nums | (1, 2, 3, 4) | ✅ 多尺度配置 |
| patch_size | 1 | ✅ Patch 大小 |
| vae_lr_schedule | "cos" | ✅ 余弦衰减 |
| optimizer_betas (VAE) | [0.5, 0.9] | ✅ AdamW betas |

#### AR (CFAP) 配置
| 参数 | 值 | 说明 |
|------|-----|------|
| ar_depth | 32 | ✅ Transformer 层数 |
| ar_embed_dim | 160 | ✅ Embedding 维度 |
| ar_num_heads | 32 | ✅ 注意力头数 |
| ar_mlp_ratio | 4.0 | ✅ MLP 倍数 |
| ar_lr | 1e-4 | ✅ 学习率 |
| ar_weight_decay | 0.05 | ✅ 权重衰减 |
| ar_dropout | 0.0 | ✅ Dropout |
| ar_attn_dropout | 0.0 | ✅ 注意力 Dropout |
| ar_drop_path_rate | 0.0 | ✅ Drop path |
| ar_label_smoothing | 0.0 | ✅ Label smoothing |
| ar_lr_schedule | "lin0" | ✅ 线性衰减到0 |
| n_obs_steps (tnobs) | 1 | ✅ 观测历史长度 |
| task_num | 8 | ✅ 任务数量 |
| task_embed_dim | 3 | ✅ 任务 embedding 维度 |

### 需要修改的配置

只有一个关键参数需要修改：**vocab_size: 512 → 1024**

原因：
- 原始 CARP 使用 1024 大小的 codebook
- 更大的 codebook 提供更丰富的动作表示
- 论文实验都基于 vocab_size=1024

## 训练超参数建议

### LIBERO 数据集规模估算
假设使用 `/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero`:
- 8个任务 (libero_spatial, libero_object, libero_goal, libero_10 可能包含多个子任务)
- 每个任务 ~50-200 episodes
- 总计约 400-1600 episodes
- 每个 episode ~50-200 步
- 总帧数估计: 20,000-320,000 帧

### Batch Size 和 Steps 建议

#### Stage 1: VAE Training
- **原始配置**: bs=500, ep=50
- **建议配置** (单 GPU):
  - batch_size: 32-64 (取决于 GPU 内存)
  - steps: 50,000-100,000 (相当于 50 epochs @ ~1000 steps/epoch)
  - save_freq: 10,000
  - eval_freq: 10,000

#### Stage 2: AR Training
- **原始配置**: bs=512, ep=200
- **建议配置** (单 GPU):
  - batch_size: 16-32
  - steps: 100,000-200,000 (相当于 100-200 epochs)
  - save_freq: 20,000
  - eval_freq: 20,000

### 其他关键参数

#### 数据加载
- num_workers: 4-8 (取决于 CPU 核数)
- 原始使用 workers=16，但 LeRobot 默认是 4

#### 优化器 (已在代码中正确设置)
- VAE: AdamW(lr=3e-4, wd=0.005, betas=[0.5, 0.9])
- AR: AdamW(lr=1e-4, wd=0.05, betas=[0.9, 0.999])

#### 学习率调度 (已在代码中正确设置)
- VAE: cosine annealing
- AR: linear decay to 0

#### Gradient Clipping (原始 CARP)
- VAE: 10.0
- AR: 2.0
- **注意**: LeRobot 的 optimizer 配置中有 grad_clip_norm 参数

## 数据集路径配置

原始 CARP 使用的数据格式:
- MimicGen HDF5 格式 (absolute action)
- 8个任务按固定顺序: coffee, hammer, mug, nut, square, stack, stackthree, threading

LeRobot 数据格式:
- LeRobotDataset (Parquet + MP4/images)
- 使用 dataset.repo_id 和 dataset.root 指定

**需要确认**: LIBERO 数据集是否已转换为 LeRobot 格式

## 正式训练命令 (下一节)

