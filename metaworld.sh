#!/bin/bash

# ================= 配置区域 =================

# 1. 定义 tmux 会话名称 (防止和之前的冲突)
SESSION_NAME="eval_meta_tasks"

# 2. 定义 Conda 环境名称
CONDA_ENV="lerobot"

# 3. 定义要并行跑的任务列表 (难度)
# 注意：这里修正了 "vey_hard" 为 "very_hard"，如果您的环境定义确实是 vey_hard 请手动修改
TASKS=("easy" "medium" "hard" "very_hard")

# 4. 定义固定变量 (所有任务共用一个模型路径)
# 模型路径
POLICY_PATH="/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/Projects/lerobot/outputs/pi05_metaworld_0123/checkpoints/020000/pretrained_model"
# 基础输出目录
BASE_OUTPUT_DIR="./eval/pi05_metaworld_20000"

# ================= 脚本逻辑 =================

# 创建一个新的 tmux session (后台运行)
tmux kill-session -t $SESSION_NAME 2>/dev/null
tmux new-session -d -s $SESSION_NAME

echo "开始启动 MetaWorld 并行评估任务 (Conda环境: $CONDA_ENV)..."
echo "使用模型: $POLICY_PATH"

# 循环遍历所有 任务难度 (Tasks)
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    
    # 自动轮询分配 GPU (0-7)
    # 比如: easy->0, medium->1, hard->2, very_hard->3
    gpu_id=$(((i) % 8))
    
    # 定义输出目录 (按任务区分文件夹，防止覆盖)
    output_dir="${BASE_OUTPUT_DIR}/${task}/"
    log_file="${output_dir}/eval_log.txt"
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 构建评估命令
    # 注意：这里 --env.task 动态使用当前的 $task
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl lerobot-eval \
        --env.type=metaworld \
        --env.task=$task \
        --eval.batch_size=5 \
        --eval.n_episodes=10 \
        --policy.path=$POLICY_PATH \
        --policy.n_action_steps=10 \
        --policy.device=cuda \
        --output_dir=${output_dir} > ${log_file} 2>&1"
        
    window_name="task_${task}"

    # 在 tmux 中创建新窗口
    # 注意：这里我们统一全部新建窗口，保留 0 号窗口作为监控或空闲窗口
    tmux new-window -t $SESSION_NAME -n "$window_name"
    target_window="$SESSION_NAME:$window_name"

    # === 执行逻辑 ===
    # 1. source bashrc
    sleep 1
    tmux send-keys -t $target_window "source ~/.bashrc" C-m
    sleep 1
    # 2. 激活环境
    tmux send-keys -t $target_window "conda activate $CONDA_ENV" C-m
    sleep 2
    # 3. 执行评估命令
    tmux send-keys -t $target_window "$cmd" C-m
    
    echo "  [+] 已启动任务: $task | GPU: $gpu_id | Log: $log_file"
    sleep 1
done

echo "---------------------------------------------------"
echo "所有 MetaWorld 任务已后台启动。"
echo "请使用命令 'tmux attach -t $SESSION_NAME' 查看。"