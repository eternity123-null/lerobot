#!/bin/bash
# 批量训练 ACT policy - 每个 task 训练一个模型
# 支持并行训练多个任务（通过指定不同 GPU）

set -e

# ========== 配置 ==========
DATASET_ROOT="/inspire/hdd/project/robot-decision/public/datasets/real_aloha1"
OUTPUT_ROOT="outputs/act_real_aloha"
LOG_DIR="logs/act_real_aloha"
BATCH_SIZE=32

# 创建日志目录
mkdir -p $LOG_DIR


# Task 列表
TASKS=(
    "place_object_in_the_plate"
    "place_the_white_mug_on_the_plate"
    "place_block_on_the_plate"
    "place_the_red_marker_in_the_pen_holder"
    "place_banana_on_the_plate"
    "stack_the_right_block_on_the_left_block"
)

# GPU 分配（可以根据需要调整）
# 格式: TASK_INDEX:GPU_ID
# 如果只有一个 GPU，所有任务会顺序执行
GPUS=(2 3 4 5 6 7)  # 可用的 GPU 列表

# ========== 训练函数 ==========
train_task() {
    local task=$1
    local gpu=$2
    local log_file="$LOG_DIR/${task}.txt"
    
    echo "=============================================="
    echo "开始训练: $task"
    echo "GPU: $gpu"
    echo "日志: $log_file"
    echo "=============================================="
    
    CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline lerobot-train \
        --dataset.repo_id=$task \
        --dataset.root=$DATASET_ROOT/$task \
        --policy.type=act \
        --output_dir=$OUTPUT_ROOT/$task \
        --job_name=act_$task \
        --policy.device=cuda \
        --wandb.enable=true \
        --wandb.mode=offline \
        --policy.push_to_hub=false \
        --batch_size=$BATCH_SIZE \
        2>&1 | tee "$log_file"
    
    echo "=============================================="
    echo "完成训练: $task"
    echo "=============================================="
}

# ========== 主逻辑 ==========
MODE=${1:-"parallel"}  # sequential 或 parallel

echo "=========================================="
echo "ACT Policy 批量训练"
echo "模式: $MODE"
echo "任务数: ${#TASKS[@]}"
echo "=========================================="

if [ "$MODE" == "parallel" ]; then
    # 并行模式：每个任务在不同 GPU 上同时运行
    echo "启动并行训练..."
    
    pids=()
    for i in "${!TASKS[@]}"; do
        task=${TASKS[$i]}
        gpu_idx=$((i % ${#GPUS[@]}))
        gpu=${GPUS[$gpu_idx]}
        
        echo "提交任务: $task -> GPU $gpu"
        train_task "$task" "$gpu" &
        pids+=($!)
        
        # 如果 GPU 用完了，等待当前批次完成
        if [ $((($i + 1) % ${#GPUS[@]})) -eq 0 ] && [ $i -lt $((${#TASKS[@]} - 1)) ]; then
            echo "等待当前批次完成..."
            for pid in "${pids[@]}"; do
                wait $pid
            done
            pids=()
        fi
    done
    
    # 等待剩余任务完成
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
else
    # 顺序模式：依次训练每个任务
    echo "启动顺序训练..."
    
    for i in "${!TASKS[@]}"; do
        task=${TASKS[$i]}
        gpu=${GPUS[0]}  # 顺序模式使用第一个 GPU
        
        echo ""
        echo "[$((i+1))/${#TASKS[@]}] 训练任务: $task"
        train_task "$task" "$gpu"
    done
fi

echo ""
echo "=========================================="
echo "所有任务训练完成!"
echo "=========================================="
echo "输出目录: $OUTPUT_ROOT"
echo "日志目录: $LOG_DIR"
for task in "${TASKS[@]}"; do
    echo "  - $task"
    echo "    模型: $OUTPUT_ROOT/$task"
    echo "    日志: $LOG_DIR/${task}.txt"
done
