#!/bin/bash

# ================= 配置区域 =================

# 1. 定义 tmux 会话名称
SESSION_NAME="eval_0122"

# 2. 定义 Conda 环境名称
CONDA_ENV="lerobot"

# 3. 定义你要评估的 Checkpoint 列表
# 在这里填入所有你想跑的 checkpoint 编号
# CHECKPOINTS=("003000" "004000" "005000" "006000" "007000" "008000" "009000" "010000")
# CHECKPOINTS=("005000" "010000" "015000" "020000" "025000" "030000" "035000" "040000")
CHECKPOINTS=("001000" "002000" "003000" "004000" "005000" "006000" "007000" "008000")
# CHECKPOINTS=("009000" "010000")
# 4. 定义路径变量
BASE_POLICY_PATH="/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/Projects/lerobot/outputs/pi05_0121/checkpoints"
BASE_OUTPUT_DIR="./eval/pi05_0121"

# ================= 脚本逻辑 =================

# 创建一个新的 tmux session (后台运行)
tmux kill-session -t $SESSION_NAME 2>/dev/null
tmux new-session -d -s $SESSION_NAME

echo "开始启动评估任务 (Conda环境: $CONDA_ENV)..."

# 循环遍历所有 Checkpoint
for i in "${!CHECKPOINTS[@]}"; do
    ckpt="${CHECKPOINTS[$i]}"
    
    # 自动轮询分配 GPU (0-7)
    gpu_id=$((i % 8))
    
    # 定义输出目录和日志
    output_dir="${BASE_OUTPUT_DIR}/${ckpt}/"
    log_file="${output_dir}/eval_log.txt"
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 构建评估命令
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id TOKENIZERS_PARALLELISM=false lerobot-eval \
        --env.type=libero \
        --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
        --eval.batch_size=5 \
        --eval.n_episodes=10 \
        --policy.path=${BASE_POLICY_PATH}/${ckpt}/pretrained_model \
        --policy.n_action_steps=10 \
        --output_dir=${output_dir} > ${log_file} 2>&1"
        
    window_name="ckpt_${ckpt}"

    # # 在 tmux 中操作
    # if [ $i -eq 0 ]; then
    #     # 第一个窗口（默认已存在）
    #     target_window="$SESSION_NAME:0"
    #     tmux rename-window -t $target_window "$window_name"
    # else
    #     # 新建窗口
    #     tmux new-window -t $SESSION_NAME -n "$window_name"
    #     target_window="$SESSION_NAME:$window_name"
    # fi
    tmux new-window -t $SESSION_NAME -n "$window_name"
    target_window="$SESSION_NAME:$window_name"

    # === 关键修改：先激活 Conda 环境，再运行命令 ===
    # 1. 先 source bashrc 确保 conda 命令可用 (防止部分机器 tmux 不加载配置)
    sleep 2
    tmux send-keys -t $target_window "source ~/.bashrc" C-m
    sleep 2
    # 2. 激活环境
    tmux send-keys -t $target_window "conda activate $CONDA_ENV" C-m
    sleep 2
    # 3. 执行评估命令
    tmux send-keys -t $target_window "$cmd" C-m
    
    echo "  [+] 已启动 Checkpoint: $ckpt | GPU: $gpu_id | Env: $CONDA_ENV"
    sleep 2
done

echo "---------------------------------------------------"
echo "所有任务已后台启动。"
echo "请使用命令 'tmux attach -t $SESSION_NAME' 查看。"



# CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false lerobot-eval         --env.type=libero         --env.task=libero_spatial,libero_object,libero_goal,libero_10         --eval.batch_size=5         --eval.n_episodes=10         --policy.path=/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/Projects/lerobot/outputs/pi0_fast0119/checkpoints/011000/pretrained_model         --policy.n_action_steps=10         --output_dir=./eval/pi0_fast0119_011000/ > ./eval/pi0_fast0119_011000/eval_log.txt 2>&1
# CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false lerobot-eval         --env.type=libero         --env.task=libero_spatial,libero_object,libero_goal,libero_10         --eval.batch_size=5         --eval.n_episodes=10         --policy.path=/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/Projects/lerobot/outputs/pi0_fast0120/checkpoints/005000/pretrained_model         --policy.n_action_steps=10         --output_dir=./eval/pi0_fast0120/005000/ > ./eval/pi0_fast0120/005000//eval_log.txt 2>&1