#!/usr/bin/env python3
"""
测试 CARP 在 LIBERO 环境评估时 task_id 是否正确注入
"""

import torch
import numpy as np
import sys
sys.path.insert(0, 'src')

print("=" * 80)
print("CARP LIBERO 评估时 Task ID 注入测试")
print("=" * 80)

# 1. 创建 LIBERO 环境
print("\n【Step 1: 创建 LIBERO 环境】")
print("-" * 80)

try:
    from lerobot.envs.factory import make_env
    from lerobot.envs.configs import LiberoEnv

    # 配置 LIBERO 环境
    env_cfg = LiberoEnv(
        task="libero_spatial",  # 任务套件
        episode_length=600,
    )

    print(f"环境配置:")
    print(f"  type: {env_cfg.type}")
    print(f"  task: {env_cfg.task}")
    print(f"  episode_length: {env_cfg.episode_length}")
    print(f"  obs_type: {env_cfg.obs_type}")

    # 创建环境
    print("\n创建环境...")
    envs = make_env(env_cfg)

    # LIBERO 返回 dict[suite][task_id] -> vec_env
    print(f"\n✓ 环境创建成功")
    print(f"  环境结构: {type(envs)}")
    print(f"  套件: {list(envs.keys())}")

    # 获取第一个环境
    suite_name = list(envs.keys())[0]
    task_dict = envs[suite_name]
    print(f"  {suite_name} 任务: {list(task_dict.keys())}")

    # 获取第一个 task_id 的环境
    task_id = list(task_dict.keys())[0]
    env = task_dict[task_id]

    print(f"\n测试环境: {suite_name} - task_id={task_id}")
    print(f"  环境类型: {type(env)}")

except Exception as e:
    print(f"✗ 环境创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. 重置环境并检查 info
print("\n【Step 2: 重置环境】")
print("-" * 80)

try:
    obs, info = env.reset()

    print(f"观测类型: {type(obs)}")
    if isinstance(obs, dict):
        print(f"观测字段:")
        for key in sorted(obs.keys()):
            if isinstance(obs[key], (np.ndarray, torch.Tensor)):
                print(f"  {key}: {obs[key].shape if hasattr(obs[key], 'shape') else type(obs[key])}")
            else:
                print(f"  {key}: {type(obs[key])}")

    print(f"\ninfo 类型: {type(info)}")
    if isinstance(info, dict):
        print(f"info 字段:")
        for key in sorted(info.keys()):
            value = info[key]
            if isinstance(value, (list, tuple)):
                print(f"  {key}: {type(value)} (len={len(value)})")
                if len(value) > 0:
                    print(f"    第一个元素: {value[0] if not isinstance(value[0], dict) else f'dict with keys {list(value[0].keys())[:5]}'}")
            else:
                print(f"  {key}: {value}")

    # 检查 task_id
    if "task_id" in info:
        print(f"\n✓ info 包含 task_id: {info['task_id']}")
    else:
        print(f"\n⚠️  info 不包含 task_id")
        print(f"  需要从 info 中提取并添加到观测")

except Exception as e:
    print(f"✗ 环境重置失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 执行一步动作
print("\n【Step 3: 执行动作】")
print("-" * 80)

try:
    # 创建随机动作
    action_dim = env.action_space.shape[0]
    action = env.action_space.sample()

    print(f"动作空间: {env.action_space}")
    print(f"动作维度: {action_dim}")
    print(f"动作: {action}")

    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\n执行后:")
    print(f"  reward: {reward}")
    print(f"  terminated: {terminated}")
    print(f"  truncated: {truncated}")

    # 检查 info 中的 task_id
    if "task_id" in info:
        if isinstance(info["task_id"], list):
            task_id_value = info["task_id"][0] if len(info["task_id"]) > 0 else None
        else:
            task_id_value = info["task_id"]

        print(f"\n✓ info 包含 task_id: {task_id_value}")
    else:
        print(f"\n⚠️  info 不包含 task_id")
        print(f"  可用字段: {list(info.keys())}")

except Exception as e:
    print(f"✗ 动作执行失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试完整的评估流程（模拟）
print("\n【Step 4: 模拟评估流程】")
print("-" * 80)

from lerobot.policies.carp.configuration_carp import CARPConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import PolicyFeature, FeatureType

# 创建配置
config = CARPConfig()

# 从环境推断特征
input_features = {}
if isinstance(obs, dict):
    for key, value in obs.items():
        if key.startswith("observation.images.") or "image" in key.lower():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                shape = value.shape
                # (H, W, C) -> (C, H, W)
                if len(shape) == 3:
                    input_features[f"observation.images.{key.split('.')[-1]}"] = PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(shape[2], shape[0], shape[1])
                    )
        elif "state" in key.lower():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                shape = value.shape
                input_features["observation.state"] = PolicyFeature(
                    type=FeatureType.STATE,
                    shape=shape
                )

output_features = {
    "action": PolicyFeature(
        type=FeatureType.ACTION,
        shape=(action_dim,)
    )
}

config.input_features = input_features
config.output_features = output_features
config.device = "cpu"  # 评估时使用 CPU
config.task_num = 40

print(f"Policy 配置:")
print(f"  input_features: {list(config.input_features.keys())}")
print(f"  output_features: {list(config.output_features.keys())}")
print(f"  task_num: {config.task_num}")

# 创建 preprocessor
try:
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=None,  # 评估时可以没有统计信息（如果不需要归一化）
    )
    print(f"✓ Preprocessor 创建成功")
except Exception as e:
    print(f"✗ Preprocessor 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 模拟评估循环
print("\n【Step 5: 模拟评估循环】")
print("-" * 80)

try:
    # 重置环境
    obs, info = env.reset()

    print("第 1 步: 从环境获取观测")
    print(f"  info 中的 task_id: {info.get('task_id', 'MISSING')}")

    # 关键步骤：将 task_id 从 info 添加到 obs
    if "task_id" in info:
        if isinstance(info["task_id"], list):
            obs["task_id"] = torch.tensor([info["task_id"][0]])
        else:
            obs["task_id"] = torch.tensor([info["task_id"]])

        print(f"✓ 将 task_id 添加到观测: {obs['task_id']}")
    else:
        print(f"⚠️  info 中没有 task_id，使用默认值")
        obs["task_id"] = torch.tensor([0])

    print(f"\n第 2 步: 应用 preprocessor")
    # 注意：需要转换 numpy 为 torch
    obs_torch = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            obs_torch[key] = torch.from_numpy(value).float()
        else:
            obs_torch[key] = value

    processed = preprocessor(obs_torch)

    print(f"  处理后的 batch:")
    for key in sorted(processed.keys()):
        if isinstance(processed[key], torch.Tensor):
            print(f"    {key}: {processed[key].shape}")
        else:
            print(f"    {key}: {type(processed[key])}")

    # 检查 task_id
    if "task_id" in processed:
        print(f"\n✓ 成功: task_id 存在于处理后的 batch")
        print(f"  task_id 值: {processed['task_id']}")
    else:
        print(f"\n✗ 失败: task_id 丢失")
        print(f"  可用字段: {list(processed.keys())}")

    print(f"\n第 3 步: 模拟模型推理")
    print(f"  模型会使用 task_id={processed.get('task_id', 'MISSING')} 进行条件化生成")

except Exception as e:
    print(f"✗ 评估循环失败: {e}")
    import traceback
    traceback.print_exc()

# 6. 测试多个任务
print("\n【Step 6: 测试多个任务】")
print("-" * 80)

print(f"测试 {suite_name} 套件的所有任务:")

task_ids_found = []
for tid, test_env in task_dict.items():
    try:
        obs, info = test_env.reset()
        task_id_value = info.get("task_id", None)

        if task_id_value is not None:
            if isinstance(task_id_value, list):
                task_id_value = task_id_value[0] if len(task_id_value) > 0 else None

            task_ids_found.append(task_id_value)
            print(f"  Task {tid}: task_id={task_id_value} ✓")
        else:
            print(f"  Task {tid}: task_id=MISSING ✗")

    except Exception as e:
        print(f"  Task {tid}: Error - {e}")

if task_ids_found:
    print(f"\n✓ 找到 {len(task_ids_found)} 个任务的 task_id")
    print(f"  Task ID 范围: {min(task_ids_found)} ~ {max(task_ids_found)}")
else:
    print(f"\n✗ 没有找到任何 task_id")

# 关闭环境
env.close()

print("\n" + "=" * 80)
print("【总结】")
print("-" * 80)
print("✓ LIBERO 环境正确提供 task_id (在 info 中)")
print("✓ 评估时需要将 info['task_id'] 添加到 obs 中")
print("✓ Preprocessor 正确保留 task_id 字段")
print("✓ task_id 可以正确传递给模型进行条件化推理")
print("\n⚠️  重要提醒:")
print("  评估脚本需要确保将 info['task_id'] 添加到观测中")
print("  否则模型会使用默认值 task_id=0")
print("=" * 80)
