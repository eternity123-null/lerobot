# CARP Policy 使用指南

CARP (Coarse-to-Fine Autoregressive Policy) 是一个两阶段训练的机器人策略，专为多任务学习设计。

## 架构概述

CARP 包含两个核心组件：

1. **MSAT (Multi-Scale Action Tokenizer)**: VQ-VAE，将连续动作离散化为多尺度 token
2. **CFAP (Coarse-to-Fine Autoregressive Predictor)**: Transformer，从粗到细地自回归生成 token

## 训练流程

### Stage 1: VAE 训练 (Multi-Scale Action Tokenization)

训练 VQ-VAE tokenizer 学习动作的多尺度离散表示。

```bash
lerobot-train \
  --dataset.repo_id=libero \
  --dataset.root=/path/to/libero/dataset \
  --policy.type=carp_vae \
  --output_dir=./outputs/carp_vae \
  --job_name=carp_vae_training \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=50000 \
  --batch_size=32 \
  --save_freq=5000 \
  --wandb.enable=true
```

**关键参数：**
- `--policy.vocab_size`: 词汇表大小 (default: 512)
- `--policy.vocab_ch`: Latent 通道数 (default: 8)
- `--policy.vch`: VAE 隐藏层通道数 (default: 2)
- `--policy.patch_nums`: 多尺度 patch 数量 (default: [1,2,3,4])
- `--policy.vae_lr`: VAE 学习率 (default: 0.0003)
- `--policy.vqbeta`: VQ commitment loss 权重 (default: 0.25)

**输出：**
- Checkpoint: `outputs/carp_vae/checkpoints/XXXXX/pretrained_model/`
- VAE 权重: `outputs/carp_vae/checkpoints/XXXXX/pretrained_model/vae_model.pt`

### Stage 2: AR 训练 (Coarse-to-Fine Autoregressive Prediction)

使用冻结的 VAE，训练 Transformer 模型自回归生成多尺度 token。

```bash
lerobot-train \
  --dataset.repo_id=libero \
  --dataset.root=/path/to/libero/dataset \
  --policy.type=carp \
  --policy.vae_checkpoint_path=./outputs/carp_vae/checkpoints/050000/pretrained_model/vae_model.pt \
  --output_dir=./outputs/carp_ar \
  --job_name=carp_ar_training \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=100000 \
  --batch_size=16 \
  --save_freq=10000 \
  --wandb.enable=true
```

**关键参数：**
- `--policy.vae_checkpoint_path`: VAE checkpoint 路径 (必须)
- `--policy.ar_depth`: Transformer 层数 (default: 32)
- `--policy.ar_embed_dim`: Embedding 维度 (default: 160)
- `--policy.ar_num_heads`: 注意力头数 (default: 32)
- `--policy.ar_lr`: AR 学习率 (default: 0.0001)
- `--policy.ar_label_smoothing`: Label smoothing (default: 0.0)
- `--policy.n_obs_steps`: 观测历史长度 (default: 1)
- `--policy.task_num`: 任务数量 (default: 8)

## 评估

在 LIBERO 环境中评估训练好的 AR 模型：

```bash
export MUJOCO_GL=egl

lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --policy.path=./outputs/carp_ar/checkpoints/100000/pretrained_model \
  --policy.n_action_steps=10 \
  --output_dir=./eval/carp_ar_100k \
  --env.max_parallel_tasks=1
```

**多任务评估：**
```bash
lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.n_episodes=50 \
  --policy.path=./outputs/carp_ar/checkpoints/100000/pretrained_model \
  --output_dir=./eval/carp_multitask
```

## 多 GPU 训练

使用 `accelerate` 进行分布式训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  $(which lerobot-train) \
  --dataset.repo_id=libero \
  --dataset.root=/path/to/libero \
  --policy.type=carp \
  --policy.vae_checkpoint_path=./vae_model.pt \
  --output_dir=./outputs/carp_multigpu \
  --batch_size=64 \
  --steps=100000 \
  --save_freq=10000 \
  --wandb.enable=true
```

## 配置参数完整列表

### 通用参数
- `n_obs_steps`: 观测历史步数 (default: 1)
- `action_horizon`: 动作预测长度 (default: 16)
- `device`: 训练设备 (cuda/cpu)

### VAE (MSAT) 参数
- `vocab_size`: 码本大小 (default: 512)
- `vocab_ch`: Latent 通道数 (default: 8)
- `vch`: VAE 通道数 (default: 2)
- `ch_mult`: 通道倍增因子 (default: [2, 4])
- `patch_nums`: 多尺度 patch 数 (default: [1,2,3,4])
- `patch_size`: Patch 大小 (default: 1)
- `vqbeta`: VQ commitment 权重 (default: 0.25)
- `vqnorm`: 使用 Z-normalization (default: True)
- `vqresi`: Residual 权重 (default: 0.5)
- `vae_dropout`: VAE dropout 率 (default: 0.0)
- `vae_lr`: VAE 学习率 (default: 0.0003)
- `vae_weight_decay`: VAE weight decay (default: 0.005)

### AR (CFAP) 参数
- `ar_depth`: Transformer 层数 (default: 32)
- `ar_embed_dim`: Embedding 维度 (default: 160)
- `ar_num_heads`: 注意力头数 (default: 32)
- `ar_mlp_ratio`: MLP 隐藏层倍数 (default: 4.0)
- `ar_dropout`: Dropout 率 (default: 0.0)
- `ar_attn_dropout`: 注意力 dropout (default: 0.0)
- `ar_drop_path_rate`: Drop path 率 (default: 0.0)
- `ar_label_smoothing`: Label smoothing (default: 0.0)
- `ar_lr`: AR 学习率 (default: 0.0001)
- `ar_weight_decay`: AR weight decay (default: 0.05)
- `task_num`: 任务数量 (default: 8)
- `task_embed_dim`: 任务 embedding 维度 (default: 3)

## 常见问题

### 1. VAE 训练不稳定
- 降低学习率: `--policy.vae_lr=0.0001`
- 调整 VQ commitment 权重: `--policy.vqbeta=0.1`
- 增加 batch size: `--batch_size=64`

### 2. AR 训练 loss 不下降
- 检查 VAE checkpoint 路径是否正确
- 降低学习率: `--policy.ar_lr=0.00005`
- 增加 warmup steps
- 尝试 label smoothing: `--policy.ar_label_smoothing=0.1`

### 3. 内存不足
- 减小 batch size
- 降低 AR 模型深度: `--policy.ar_depth=16`
- 减少注意力头数: `--policy.ar_num_heads=16`
- 使用梯度 checkpointing (需在代码中启用)

### 4. 评估性能不佳
- 增加训练步数
- 使用更大的模型 (ar_depth, ar_embed_dim)
- 检查数据集质量
- 尝试不同的 action_horizon 值
- 调整 n_action_steps 参数

## 论文参考

详细的技术说明请参考：
- 论文: [CARP: Coarse-to-Fine Autoregressive Policy for Robotics]
- 位置: `carp/Coarse-to-Fine Autoregressive Networks for Videos.pdf`
