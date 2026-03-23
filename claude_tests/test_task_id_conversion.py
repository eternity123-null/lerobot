#!/usr/bin/env python3
"""
测试 task_id 的完整数据流（numpy → tensor）
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation

print("=" * 80)
print("Task ID 数据类型转换测试")
print("=" * 80)

# 模拟从环境获取的观测（包含 task_id）
mock_observation = {
    "pixels": {
        "image": np.random.randint(0, 255, (1, 360, 360, 3), dtype=np.uint8),
    },
    "robot_state": {
        "eef": {
            "pos": np.random.randn(1, 3).astype(np.float32),
        },
        "joint": {
            "pos": np.random.randn(1, 7).astype(np.float32),
        },
    },
    "task_id": np.array([15], dtype=np.int64),  # ← numpy array
}

print("\n1. 输入观测")
print(f"   task_id 类型: {type(mock_observation['task_id'])}")
print(f"   task_id 值: {mock_observation['task_id']}")
print(f"   task_id dtype: {mock_observation['task_id'].dtype}")

# 应用 preprocess_observation
processed = preprocess_observation(mock_observation)

print("\n2. 预处理后")
if "task_id" in processed:
    task_id = processed["task_id"]
    print(f"   task_id 类型: {type(task_id)}")
    print(f"   task_id 值: {task_id}")
    print(f"   task_id dtype: {task_id.dtype}")
    print(f"   task_id shape: {task_id.shape}")

    # 验证类型正确
    if isinstance(task_id, torch.Tensor):
        print(f"\n   ✓ 类型正确: torch.Tensor")
        if task_id.dtype == torch.long or task_id.dtype == torch.int64:
            print(f"   ✓ dtype 正确: {task_id.dtype}")
        else:
            print(f"   ✗ dtype 错误: {task_id.dtype}，期望 torch.long")
    else:
        print(f"\n   ✗ 类型错误: {type(task_id)}，期望 torch.Tensor")
else:
    print(f"   ✗ task_id 字段丢失！")

# 模拟 nn.Embedding 调用
print("\n3. 模拟 task_embed 调用")
try:
    task_embed = torch.nn.Embedding(40, 128)  # 40 tasks, 128 dim
    embedded = task_embed(processed["task_id"])
    print(f"   ✓ task_embed 调用成功")
    print(f"   embedded shape: {embedded.shape}")
except Exception as e:
    print(f"   ✗ task_embed 调用失败: {e}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
