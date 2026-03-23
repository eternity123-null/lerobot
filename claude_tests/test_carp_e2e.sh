#!/bin/bash
# CARP 端到端训练和评估测试脚本

set -e  # 遇到错误立即退出

echo "==========================================="
echo "CARP 端到端测试"
echo "==========================================="

# 配置
DATASET_ROOT="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero"
DATASET_REPO_ID="libero"
OUTPUT_BASE="./outputs/test_carp_e2e"
DEVICE="cuda"

# 清理之前的输出
if [ -d "$OUTPUT_BASE" ]; then
    echo "清理之前的输出目录: $OUTPUT_BASE"
    rm -rf "$OUTPUT_BASE"
fi

# ============================================
# Stage 1: VAE 训练
# ============================================
echo ""
echo "==========================================="
echo "Stage 1: 训练 CARP VAE (Multi-Scale Action Tokenizer)"
echo "==========================================="

VAE_OUTPUT_DIR="$OUTPUT_BASE/vae"

CUDA_VISIBLE_DEVICES=0 lerobot-train \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --policy.type=carp_vae \
    --output_dir="$VAE_OUTPUT_DIR" \
    --job_name=carp_vae_test \
    --policy.device="$DEVICE" \
    --policy.push_to_hub=false \
    --steps=100 \
    --batch_size=4 \
    --save_freq=50 \
    --eval_freq=50 \
    --log_freq=10 \
    --wandb.enable=false

# 检查 VAE checkpoint 是否生成
VAE_CHECKPOINT="$VAE_OUTPUT_DIR/checkpoints/000050/pretrained_model"
if [ ! -d "$VAE_CHECKPOINT" ]; then
    echo "✗ VAE checkpoint 未生成: $VAE_CHECKPOINT"
    exit 1
fi

echo "✓ VAE 训练完成，checkpoint 保存在: $VAE_CHECKPOINT"

# ============================================
# Stage 2: AR 训练
# ============================================
echo ""
echo "==========================================="
echo "Stage 2: 训练 CARP AR (Coarse-to-Fine Autoregressive Prediction)"
echo "==========================================="

AR_OUTPUT_DIR="$OUTPUT_BASE/ar"

CUDA_VISIBLE_DEVICES=0 lerobot-train \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --policy.type=carp \
    --policy.vae_checkpoint_path="$VAE_CHECKPOINT/vae_model.pt" \
    --output_dir="$AR_OUTPUT_DIR" \
    --job_name=carp_ar_test \
    --policy.device="$DEVICE" \
    --policy.push_to_hub=false \
    --steps=100 \
    --batch_size=2 \
    --save_freq=50 \
    --eval_freq=50 \
    --log_freq=10 \
    --wandb.enable=false

# 检查 AR checkpoint 是否生成
AR_CHECKPOINT="$AR_OUTPUT_DIR/checkpoints/000050/pretrained_model"
if [ ! -d "$AR_CHECKPOINT" ]; then
    echo "✗ AR checkpoint 未生成: $AR_CHECKPOINT"
    exit 1
fi

echo "✓ AR 训练完成，checkpoint 保存在: $AR_CHECKPOINT"

# ============================================
# Stage 3: LIBERO 评估
# ============================================
echo ""
echo "==========================================="
echo "Stage 3: LIBERO 环境评估"
echo "==========================================="

export MUJOCO_GL=egl

EVAL_OUTPUT_DIR="$OUTPUT_BASE/eval"

CUDA_VISIBLE_DEVICES=0 lerobot-eval \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=1 \
    --eval.n_episodes=2 \
    --policy.path="$AR_CHECKPOINT" \
    --policy.n_action_steps=10 \
    --output_dir="$EVAL_OUTPUT_DIR" \
    --env.max_parallel_tasks=1

echo ""
echo "==========================================="
echo "✓ 所有测试完成！"
echo "==========================================="
echo "VAE checkpoint: $VAE_CHECKPOINT"
echo "AR checkpoint: $AR_CHECKPOINT"
echo "评估结果: $EVAL_OUTPUT_DIR"
echo "==========================================="
