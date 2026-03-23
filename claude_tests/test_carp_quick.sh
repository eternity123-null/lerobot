#!/bin/bash
# CARP 快速测试脚本 - 验证训练流程

set -e

echo "==========================================="
echo "CARP 快速训练测试"
echo "==========================================="

# 配置
DATASET_ROOT="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero"
DATASET_REPO_ID="libero"
OUTPUT_DIR="./outputs/test_carp_quick"
DEVICE="cuda"

# 清理之前的输出
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# ============================================
# Stage 1: VAE 训练（快速测试）
# ============================================
echo ""
echo "Stage 1: 测试 VAE 训练..."

VAE_DIR="$OUTPUT_DIR/vae"

CUDA_VISIBLE_DEVICES=0 lerobot-train \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --policy.type=carp_vae \
    --output_dir="$VAE_DIR" \
    --job_name=carp_vae_quick \
    --policy.device="$DEVICE" \
    --policy.push_to_hub=false \
    --steps=10 \
    --batch_size=2 \
    --save_freq=10 \
    --log_freq=2 \
    --wandb.enable=false 2>&1 | tee "$OUTPUT_DIR/vae_train.log"

# 检查 VAE checkpoint
VAE_CHECKPOINT="$VAE_DIR/checkpoints/000010/pretrained_model"
if [ ! -f "$VAE_CHECKPOINT/vae_model.pt" ]; then
    echo "✗ VAE checkpoint 未找到: $VAE_CHECKPOINT/vae_model.pt"
    echo "目录内容："
    ls -la "$VAE_CHECKPOINT" || echo "目录不存在"
    exit 1
fi

echo "✓ VAE 训练成功"

# ============================================
# Stage 2: AR 训练（快速测试）
# ============================================
echo ""
echo "Stage 2: 测试 AR 训练..."

AR_DIR="$OUTPUT_DIR/ar"

CUDA_VISIBLE_DEVICES=0 lerobot-train \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --policy.type=carp \
    --policy.vae_checkpoint_path="$VAE_CHECKPOINT/vae_model.pt" \
    --output_dir="$AR_DIR" \
    --job_name=carp_ar_quick \
    --policy.device="$DEVICE" \
    --policy.push_to_hub=false \
    --steps=10 \
    --batch_size=2 \
    --save_freq=10 \
    --log_freq=2 \
    --wandb.enable=false 2>&1 | tee "$OUTPUT_DIR/ar_train.log"

# 检查 AR checkpoint
AR_CHECKPOINT="$AR_DIR/checkpoints/000010/pretrained_model"
if [ ! -d "$AR_CHECKPOINT" ]; then
    echo "✗ AR checkpoint 未生成: $AR_CHECKPOINT"
    exit 1
fi

echo "✓ AR 训练成功"

echo ""
echo "==========================================="
echo "✓ 快速测试完成！"
echo "==========================================="
echo "VAE: $VAE_CHECKPOINT"
echo "AR:  $AR_CHECKPOINT"
echo "==========================================="
