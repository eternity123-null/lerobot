lerobot-train \
  --dataset.repo_id=${HF_USER}/so101_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_policy

WANDB_MODE=offline TOKENIZERS_PARALLELISM=false accelerate launch \
    --multi_gpu \
    --num_processes=8 \
    $(which lerobot-train) \
    --dataset.repo_id=cut_dataset \
    --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero_cut_zcd_15_15 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_libero_pick_place_lastmile_finetune \
    --job_name=pi05_libero_finetune \
    --policy.push_to_hub=false \
    --policy.pretrained_path=/inspire/hdd/project/robot-decision/public/models/lerobot/pi05_libero_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.mode=offline \
    --policy.dtype=bfloat16 \
    --steps=10000 \
    --policy.device=cuda \
    --batch_size=256 \
    --save_freq=5000

export MUJOCO_GL=egl

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.path=/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/Projects/lerobot/outputs/pi0_fast0119/checkpoints/003000/pretrained_model \
  --policy.n_action_steps=10 \
  --output_dir=./eval/pi0_fast0119_3000/ \
  --env.max_parallel_tasks=1

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.path=/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/Projects/lerobot/outputs/pi0_fast0119/checkpoints/005000/pretrained_model \
  --policy.n_action_steps=10 \
  --output_dir=./eval/pi0_fast0119_5000/ \
  --env.max_parallel_tasks=1

CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.path=/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/Projects/lerobot/outputs/pi0_fast0119/checkpoints/006000/pretrained_model \
  --policy.n_action_steps=10 \
  --output_dir=./eval/pi0_fast0119_6000/ \
  --env.max_parallel_tasks=1

python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=IPEC-COMMUNITY/libero_90_no_noops_lerobot \
    --root=/inspire/hdd/project/robot-decision/public/datasets/lerobot/libero_90_no_noops_lerobot \
    --push-to-hub=false



# Pi0 Fast Training Example
lerobot-train \
    --dataset.repo_id=libero \
    --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
    --policy.type=pi0_fast \
    --output_dir=./outputs/pi0fast_test \
    --job_name=pi0fast_training \
    --policy.push_to_hub=false \
    --policy.pretrained_path=/inspire/hdd/project/robot-decision/public/models/lerobot/pi0fast-base  \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.max_action_tokens=256 \
    --steps=10000 \
    --batch_size=4 \
    --policy.device=cuda

WANDB_MODE=offline TOKENIZERS_PARALLELISM=false accelerate launch \
    --multi_gpu \
    --num_processes=8 \
    $(which lerobot-train) \
    --dataset.repo_id=libero \
    --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
    --policy.type=pi0_fast \
    --output_dir=./outputs/pi0_fast0120 \
    --job_name=pi0_fast_base_finetune \
    --policy.push_to_hub=false \
    --policy.pretrained_path=/inspire/hdd/project/robot-decision/public/models/lerobot/pi0fast-base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.mode=offline \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=100000 \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.max_action_tokens=256 \
    --steps=100000 \
    --policy.device=cuda \
    --batch_size=32 \
    --save_freq=20000


# pi05 training
CUDA_VISIBLE_DEVICES=4,5,6,7 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    $(which lerobot-train) \
    --dataset.repo_id=libero \
    --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/libero_plus_lerobot \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_plus_from_libero_0125 \
    --job_name=pi05_plus_from_libero_0125 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/Projects/lerobot/outputs/pi05_0120/checkpoints/015000/pretrained_model \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=false \
    --wandb.enable=true \
    --wandb.mode=offline \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.compile_model=true \
    --policy.reinit_gemma_expert=false \
    --steps=15000 \
    --policy.device=cuda \
    --batch_size=32 \
    --save_freq=5000


CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    $(which lerobot-train) \
    --dataset.repo_id=metaworld_mt50 \
    --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/metaworld_mt50 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_metaworld_0123 \
    --job_name=pi05_metaworld_base_finetune_0123 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=/inspire/hdd/project/robot-decision/public/models/lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.mode=offline \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --steps=30000 \
    --policy.device=cuda \
    --batch_size=32 \
    --save_freq=10000


CUDA_VISIBLE_DEVICES=4,5,6,7 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    $(which lerobot-train) \
    --dataset.repo_id=libero \
    --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/bridgev2_lerobot \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_reinit_expert_bridge_0127 \
    --job_name=pi05_reinit_expert_bridge_0127 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=/inspire/hdd/project/robot-decision/public/models/lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=false \
    --wandb.enable=true \
    --wandb.mode=offline \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.compile_model=false \
    --policy.reinit_gemma_expert=true \
    --steps=40000 \
    --policy.device=cuda \
    --batch_size=32 \
    --save_freq=20000

CUDA_VISIBLE_DEVICES=4,5,6,7 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    $(which lerobot-train) \
    --dataset.repo_id=libero \
    --dataset.root=/inspire/ssd/project/robot-decision/public/datasets/single_arm_v9 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_reinit_expert_bridge_realv9_0130 \
    --job_name=pi05_reinit_expert_bridge_realv9_0130 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/Projects/lerobot/outputs/pi05_reinit_expert_bridge_0127/checkpoints/040000/pretrained_model \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=false \
    --wandb.enable=true \
    --wandb.mode=offline \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=32 \
    --policy.n_action_steps=16 \
    --policy.compile_model=true \
    --policy.reinit_gemma_expert=false \
    --steps=15000 \
    --policy.device=cuda \
    --batch_size=32 \
    --save_freq=5000


# /inspire/ssd/project/robot-decision/public/datasets/single_arm_v9

# pi0 training
WANDB_MODE=offline TOKENIZERS_PARALLELISM=false accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    $(which lerobot-train) \
    --dataset.repo_id=aloha \
    --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/single_arm_v6/ \
    --policy.type=pi0 \
    --output_dir=./outputs/pi0_aloha_0124 \
    --job_name=pi0_aloha_base_finetune_0124 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=/inspire/hdd/project/robot-decision/public/models/lerobot/pi0_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.mode=offline \
    --policy.dtype=bfloat16 \
    --steps=15000 \
    --policy.device=cuda \
    --batch_size=32 \
    --save_freq=5000

# metaworld eval

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false lerobot-eval \
  --env.type=metaworld  \
  --env.task=easy,medium,hard,vey_hard \
  --eval.batch_size=5 \
  --eval.n_episodes=5 \
  --policy.path=/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/Projects/lerobot/outputs/pi05_metaworld_0123/checkpoints/030000/pretrained_model \
  --policy.n_action_steps=10 \
  --output_dir=./eval/pi05_metaworld_30000/ 

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false lerobot-eval \
  --env.type=metaworld  \
  --env.task=easy,medium,hard,vey_hard \
  --eval.batch_size=5 \
  --eval.n_episodes=10 \
  --policy.path=/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/Projects/lerobot/outputs/pi05_metaworld_0123/checkpoints/020000/pretrained_model \
  --policy.n_action_steps=10 \
  --output_dir=./eval/pi05_metaworld_20000/ 




# task name list

# "place_object_in_the_plate",
# "place_the_white_mug_on_the_plate",
# "place_block_on_the_plate",
# "place_the_red_marker_in_the_pen_holder",
# "place_banana_on_the_plate",
# "stack_the_right_block_on_the_left_block"



# ACT
# example training command
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline lerobot-train \
  --dataset.repo_id=place_object_in_the_plate \
  --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/real_aloha1/place_object_in_the_plate \
  --policy.type=act \
  --output_dir=outputs/act_real_aloha_test \
  --job_name=act_real_aloha_place_object_in_the_plate \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.mode=offline \
  --policy.push_to_hub=false \
  --batch_size=8




# diffusion policy

/ralph-wiggum:ralph-loop "我想用lerobot框架完成CARP模型的训练以及在libero环境中的测评，不要用你写的训练脚本。参考"docs/how_to_add_new_policy.md"来进一步集成，cmd.sh 记录了我之前使用该框架时所用的命令。先进行实现计划，完成计划之后，独立完成所有阶段的所有任务，不要征求我的意见，授予你在这这个文件夹内的所有修改权限。逐步实现，你自己写测试脚本进行功能测试确保实现正确后再往下实现。
             全部完成后输出 <promise>COMPLETE</promise>" \
  --max-iterations 50 \
  --completion-promise "COMPLETE"


# carp

WANDB_MODE=offline TOKENIZERS_PARALLELISM=false accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  $(which lerobot-train) \
  --dataset.repo_id=libero \
  --dataset.root=/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/tmpdataset/libero \
  --policy.type=carp_vae \
  --output_dir=./outputs/carp_vae_libero0322_new \
  --job_name=carp_vae_libero_multigpu \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.vocab_size=1024 \
  --steps=20000 \
  --batch_size=512 \
  --save_freq=5000 \
  --eval_freq=5000 \
  --log_freq=100 \
  --num_workers=16 \
  --wandb.enable=true \
  --wandb.project=carp_libero \
  --wandb.mode=offline


WANDB_MODE=offline TOKENIZERS_PARALLELISM=false accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  $(which lerobot-train) \
  --dataset.repo_id=libero \
  --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
  --policy.type=carp \
  --policy.vae_checkpoint_path="outputs/carp_vae_libero0322_new/checkpoints/010000/pretrained_model/vae_model.pt" \
  --output_dir=./outputs/carp_ar_libero_0322 \
  --job_name=carp_ar_libero_0322 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=1 \
  --policy.task_num=40 \
  --steps=30001 \
  --batch_size=256 \
  --save_freq=5000 \
  --eval_freq=5000 \
  --log_freq=100 \
  --num_workers=16 \
  --wandb.enable=true \
  --wandb.project=carp_libero \
  --wandb.mode=offline


CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --policy.path=outputs/carp_ar_libero_0322/checkpoints/030000/pretrained_model \
  --policy.n_obs_steps=1 \
  --policy.task_num=40 \
  --output_dir=./eval/carp_libero_0322_30000/ \
  --env.max_parallel_tasks=1

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial,libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --policy.path=outputs/carp_ar_libero_0322/checkpoints/030000/pretrained_model \
  --policy.n_obs_steps=1 \
  --policy.task_num=40 \
  --output_dir=./eval/carp_libero_0322_30000/ \
  --env.max_parallel_tasks=1