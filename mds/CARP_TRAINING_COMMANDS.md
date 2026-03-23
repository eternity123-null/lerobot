# CARP 正式训练命令

## 数据集信息

**LIBERO 数据集统计**:
- 路径: `/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero`
- Total episodes: 1,693
- Total frames: 273,465
- Total tasks: 40
- Action dimension: 7 (Panda 机器人: 6 DoF + gripper)
- State dimension: 8
- FPS: 10.0
- 平均 episode 长度: ~161 frames

**训练规模估算**:
- 假设 batch_size=32, 每个 epoch ≈ 273,465/32/16 ≈ 534 steps (action_horizon=16)
- 50 epochs ≈ 26,700 steps
- 100 epochs ≈ 53,400 steps

## Stage 1: VAE (Multi-Scale Action Tokenizer) 训练

### 推荐配置

**单 GPU (推荐)**:
```bash
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=libero \
  --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
  --policy.type=carp_vae \
  --output_dir=./outputs/carp_vae_libero \
  --job_name=carp_vae_libero \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.vocab_size=1024 \
  --policy.vocab_ch=8 \
  --policy.vch=2 \
  --policy.vqbeta=0.25 \
  --policy.vqnorm=true \
  --policy.action_horizon=16 \
  --steps=50000 \
  --batch_size=32 \
  --save_freq=5000 \
  --eval_freq=5000 \
  --log_freq=100 \
  --num_workers=4 \
  --wandb.enable=false
```

**多 GPU (4卡加速)**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_MODE=offline \
TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  $(which lerobot-train) \
  --dataset.repo_id=libero \
  --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
  --policy.type=carp_vae \
  --output_dir=./outputs/carp_vae_libero_multigpu \
  --job_name=carp_vae_libero_multigpu \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.vocab_size=1024 \
  --steps=50000 \
  --batch_size=128 \
  --save_freq=5000 \
  --eval_freq=5000 \
  --log_freq=100 \
  --num_workers=8 \
  --wandb.enable=true \
  --wandb.project=carp_libero
```

### 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `policy.vocab_size` | 1024 | ⚠️ **必须指定** - 与原始 CARP 一致 |
| `policy.vocab_ch` | 8 | Latent 通道数 (默认值正确) |
| `policy.vch` | 2 | VAE 基础通道数 (默认值正确) |
| `policy.vqbeta` | 0.25 | VQ commitment 权重 (默认值正确) |
| `policy.vqnorm` | true | 使用 cosine similarity (默认值正确) |
| `policy.action_horizon` | 16 | 动作序列长度 (默认值正确) |
| `steps` | 50000 | 约 50 epochs @ 1000 steps/epoch |
| `batch_size` | 32 (单GPU) / 128 (4GPU) | 实际每GPU batch_size = 总batch/GPU数 |
| `save_freq` | 5000 | 每 5000 steps 保存 checkpoint |
| `eval_freq` | 5000 | 每 5000 steps 评估 |

### 预期输出

- Checkpoint 目录: `./outputs/carp_vae_libero/checkpoints/`
- 关键 checkpoint: `checkpoints/050000/pretrained_model/vae_model.pt` ← **用于 Stage 2**
- 训练日志: `./outputs/carp_vae_libero/train.log`
- TensorBoard: `./outputs/carp_vae_libero/logs/`

### 监控指标

- **recon_loss**: 重建损失，应逐渐下降 (目标: < 0.5)
- **vq_loss**: VQ commitment 损失
- **usage_scale1-4**: Codebook 使用率 (应该 > 80%)
- **grad_norm**: 梯度范数 (应稳定)

---

## Stage 2: AR (Coarse-to-Fine Autoregressive Predictor) 训练

### 推荐配置

**单 GPU (推荐)**:
```bash
# 设置 VAE checkpoint 路径
VAE_CKPT="./outputs/carp_vae_libero/checkpoints/050000/pretrained_model/vae_model.pt"

CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=libero \
  --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
  --policy.type=carp \
  --policy.vae_checkpoint_path="$VAE_CKPT" \
  --output_dir=./outputs/carp_ar_libero \
  --job_name=carp_ar_libero \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.vocab_size=1024 \
  --policy.ar_depth=32 \
  --policy.ar_embed_dim=160 \
  --policy.ar_num_heads=32 \
  --policy.n_obs_steps=1 \
  --policy.action_horizon=16 \
  --policy.task_num=40 \
  --steps=100000 \
  --batch_size=16 \
  --save_freq=10000 \
  --eval_freq=10000 \
  --log_freq=100 \
  --num_workers=4 \
  --wandb.enable=false
```

**多 GPU (4卡加速)**:
```bash
# 设置 VAE checkpoint 路径
VAE_CKPT="./outputs/carp_vae_libero_multigpu/checkpoints/050000/pretrained_model/vae_model.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_MODE=offline \
TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  $(which lerobot-train) \
  --dataset.repo_id=libero \
  --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
  --policy.type=carp \
  --policy.vae_checkpoint_path="$VAE_CKPT" \
  --output_dir=./outputs/carp_ar_libero_multigpu \
  --job_name=carp_ar_libero_multigpu \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.vocab_size=1024 \
  --policy.ar_depth=32 \
  --policy.ar_embed_dim=160 \
  --policy.n_obs_steps=1 \
  --policy.task_num=40 \
  --steps=100000 \
  --batch_size=64 \
  --save_freq=10000 \
  --eval_freq=10000 \
  --log_freq=100 \
  --num_workers=8 \
  --wandb.enable=true \
  --wandb.project=carp_libero
```

### 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `policy.vae_checkpoint_path` | path/to/vae_model.pt | ⚠️ **必须指定** - Stage 1 的输出 |
| `policy.vocab_size` | 1024 | ⚠️ **必须与 VAE 一致** |
| `policy.ar_depth` | 32 | Transformer 层数 (默认值正确) |
| `policy.ar_embed_dim` | 160 | Embedding 维度 (默认值正确) |
| `policy.ar_num_heads` | 32 | 注意力头数 (默认值正确) |
| `policy.n_obs_steps` | 1 | 观测历史长度 (默认值正确) |
| `policy.task_num` | 40 | ⚠️ **必须指定** - LIBERO 有 40 个任务 |
| `steps` | 100000 | 约 100 epochs |
| `batch_size` | 16 (单GPU) / 64 (4GPU) | AR 模型较大，batch size 较小 |

### 预期输出

- Checkpoint 目录: `./outputs/carp_ar_libero/checkpoints/`
- 最终模型: `checkpoints/100000/pretrained_model/` ← **用于评估**
- 训练日志: `./outputs/carp_ar_libero/train.log`
- TensorBoard: `./outputs/carp_ar_libero/logs/`

### 监控指标

- **loss**: 交叉熵损失，应逐渐下降 (目标: < 2.0)
- **accuracy**: Token 预测准确率 (目标: > 70%)
- **loss_scale1-4**: 各尺度的损失
- **grad_norm**: 梯度范数 (应稳定，< 100)

---

## Stage 3: LIBERO 环境评估

### 单任务评估

```bash
export MUJOCO_GL=egl

CUDA_VISIBLE_DEVICES=0 lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --policy.path=./outputs/carp_ar_libero/checkpoints/100000/pretrained_model \
  --policy.n_action_steps=10 \
  --output_dir=./eval/carp_ar_libero_spatial \
  --env.max_parallel_tasks=1
```

### 多任务评估

```bash
export MUJOCO_GL=egl

CUDA_VISIBLE_DEVICES=0 lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --policy.path=./outputs/carp_ar_libero/checkpoints/100000/pretrained_model \
  --policy.n_action_steps=10 \
  --output_dir=./eval/carp_ar_libero_multitask \
  --env.max_parallel_tasks=1
```

### 评估不同 checkpoints

```bash
# 评估多个 checkpoint 找到最佳模型
for STEP in 050000 060000 070000 080000 090000 100000; do
  CUDA_VISIBLE_DEVICES=0 lerobot-eval \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.n_episodes=20 \
    --policy.path=./outputs/carp_ar_libero/checkpoints/${STEP}/pretrained_model \
    --output_dir=./eval/carp_ar_libero_step${STEP}
done
```

### 预期结果

- **成功率**: 期望 > 60% (取决于任务难度)
- **评估日志**: `./eval/carp_ar_libero_*/eval_info.json`
- **视频录制**: `./eval/carp_ar_libero_*/videos/`

---

## 完整训练流程脚本

### 自动化两阶段训练

```bash
#!/bin/bash
# carp_full_training.sh

set -e

echo "=========================================="
echo "CARP 完整训练流程 - LIBERO 数据集"
echo "=========================================="

# 配置
DATASET_ROOT="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero"
DATASET_REPO="libero"
OUTPUT_BASE="./outputs/carp_libero_$(date +%Y%m%d_%H%M%S)"
GPU_ID=0

# Stage 1: VAE 训练
echo ""
echo "Stage 1: 训练 VAE (Multi-Scale Action Tokenizer)"
echo "------------------------------------------"

VAE_OUTPUT="${OUTPUT_BASE}/vae"

CUDA_VISIBLE_DEVICES=$GPU_ID lerobot-train \
  --dataset.repo_id=$DATASET_REPO \
  --dataset.root=$DATASET_ROOT \
  --policy.type=carp_vae \
  --output_dir=$VAE_OUTPUT \
  --job_name=carp_vae \
  --policy.device=cuda \
  --policy.vocab_size=1024 \
  --steps=50000 \
  --batch_size=32 \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=4

VAE_CKPT="${VAE_OUTPUT}/checkpoints/050000/pretrained_model/vae_model.pt"

if [ ! -f "$VAE_CKPT" ]; then
    echo "✗ VAE checkpoint 未找到: $VAE_CKPT"
    exit 1
fi

echo "✓ VAE 训练完成: $VAE_CKPT"

# Stage 2: AR 训练
echo ""
echo "Stage 2: 训练 AR (Coarse-to-Fine Autoregressive)"
echo "------------------------------------------"

AR_OUTPUT="${OUTPUT_BASE}/ar"

CUDA_VISIBLE_DEVICES=$GPU_ID lerobot-train \
  --dataset.repo_id=$DATASET_REPO \
  --dataset.root=$DATASET_ROOT \
  --policy.type=carp \
  --policy.vae_checkpoint_path=$VAE_CKPT \
  --output_dir=$AR_OUTPUT \
  --job_name=carp_ar \
  --policy.device=cuda \
  --policy.vocab_size=1024 \
  --policy.task_num=40 \
  --steps=100000 \
  --batch_size=16 \
  --save_freq=10000 \
  --log_freq=100 \
  --num_workers=4

AR_CKPT="${AR_OUTPUT}/checkpoints/100000/pretrained_model"

if [ ! -d "$AR_CKPT" ]; then
    echo "✗ AR checkpoint 未找到: $AR_CKPT"
    exit 1
fi

echo "✓ AR 训练完成: $AR_CKPT"

# Stage 3: 评估
echo ""
echo "Stage 3: LIBERO 环境评估"
echo "------------------------------------------"

export MUJOCO_GL=egl

CUDA_VISIBLE_DEVICES=$GPU_ID lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=50 \
  --policy.path=$AR_CKPT \
  --output_dir=${OUTPUT_BASE}/eval

echo ""
echo "=========================================="
echo "✓ 完整训练流程完成！"
echo "=========================================="
echo "VAE: $VAE_CKPT"
echo "AR:  $AR_CKPT"
echo "评估: ${OUTPUT_BASE}/eval"
echo "=========================================="
```

保存为 `carp_full_training.sh`，然后运行：
```bash
chmod +x carp_full_training.sh
bash carp_full_training.sh
```

---

## 训练时间估算

**硬件**: NVIDIA GPU (如 RTX 3090/4090, A100)

### Stage 1: VAE (50,000 steps)
- 单 GPU (batch_size=32): ~8-12 小时
- 4 GPU (batch_size=128): ~2-3 小时

### Stage 2: AR (100,000 steps)
- 单 GPU (batch_size=16): ~20-30 小时
- 4 GPU (batch_size=64): ~5-8 小时

### Stage 3: Evaluation (50 episodes)
- 单任务: ~30-60 分钟
- 4 任务集: ~2-4 小时

**总计 (单 GPU)**: 约 30-45 小时
**总计 (4 GPU)**: 约 8-12 小时

---

## 故障排除

### 常见问题

#### 1. OOM (Out of Memory)
- 减小 batch_size: 32→16 (VAE), 16→8 (AR)
- 减小 ar_depth: 32→24 或 16
- 减小 ar_num_heads: 32→16

#### 2. VAE codebook collapse
- 检查 usage_scale* 指标
- 增加 vqbeta: 0.25→0.5
- 确保 vqnorm=true

#### 3. AR loss 不下降
- 确认 VAE checkpoint 正确加载
- 检查 vocab_size 是否一致
- 尝试降低学习率: 1e-4→5e-5

#### 4. Task_num 不匹配
- 确认数据集的实际任务数
- LIBERO 有 40 个任务，不是 8 个
- 使用 `--policy.task_num=40`

---

## 推荐使用命令

**快速开始 (单 GPU, 适合测试)**:
```bash
# Stage 1
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --policy.type=carp_vae \
  --dataset.repo_id=libero \
  --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
  --output_dir=./outputs/carp_vae \
  --policy.vocab_size=1024 \
  --steps=50000 \
  --batch_size=32 \
  --save_freq=5000

# Stage 2
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --policy.type=carp \
  --dataset.repo_id=libero \
  --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
  --policy.vae_checkpoint_path=./outputs/carp_vae/checkpoints/050000/pretrained_model/vae_model.pt \
  --output_dir=./outputs/carp_ar \
  --policy.vocab_size=1024 \
  --policy.task_num=40 \
  --steps=100000 \
  --batch_size=16 \
  --save_freq=10000
```

**生产环境 (多 GPU, 推荐)**:
使用上面的"多 GPU"配置或"完整训练流程脚本"。
