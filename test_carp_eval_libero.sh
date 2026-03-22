#!/bin/bash
# CARP LIBERO 评估测试

set -e

echo "==========================================="
echo "CARP LIBERO 评估测试"
echo "==========================================="

# 使用训练好的 checkpoint
AR_CHECKPOINT="./outputs/test_carp_quick/ar/checkpoints/000010/pretrained_model"

if [ ! -d "$AR_CHECKPOINT" ]; then
    echo "✗ AR checkpoint 不存在: $AR_CHECKPOINT"
    echo "请先运行 test_carp_quick.sh 进行训练"
    exit 1
fi

echo "使用 checkpoint: $AR_CHECKPOINT"

# 设置环境变量
export MUJOCO_GL=egl

# 评估输出目录
EVAL_DIR="./outputs/test_carp_quick/eval"
mkdir -p "$EVAL_DIR"

echo ""
echo "开始评估（2 个 episodes，快速测试）..."

CUDA_VISIBLE_DEVICES=0 lerobot-eval \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=1 \
    --eval.n_episodes=2 \
    --policy.path="$AR_CHECKPOINT" \
    --policy.n_action_steps=10 \
    --output_dir="$EVAL_DIR" \
    --env.max_parallel_tasks=1

echo ""
echo "==========================================="
echo "✓ 评估完成！"
echo "==========================================="
echo "结果保存在: $EVAL_DIR"
echo "==========================================="

# 显示评估结果
if [ -f "$EVAL_DIR/eval_info.json" ]; then
    echo ""
    echo "评估结果摘要:"
    cat "$EVAL_DIR/eval_info.json" | python -m json.tool | head -20
fi
