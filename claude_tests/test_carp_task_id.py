#!/usr/bin/env python3
"""
测试 CARP Processor 是否正确处理 task_index → task_id 重命名
"""

import torch
import sys
sys.path.insert(0, 'src')

from lerobot.policies.carp.configuration_carp import CARPConfig
from lerobot.policies.carp.processor_carp import make_carp_pre_post_processors
from lerobot.configs.types import PolicyFeature, FeatureType

print("=" * 80)
print("CARP Processor Task ID 测试")
print("=" * 80)

# 创建配置
config = CARPConfig()
config.input_features = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
}
config.output_features = {
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
}
config.device = "cpu"
config.task_num = 40

# 创建假的数据集统计
dataset_stats = {
    "observation.state": {
        "mean": torch.zeros(8),
        "std": torch.ones(8),
    },
    "action": {
        "mean": torch.zeros(7),
        "std": torch.ones(7),
    },
}

# 创建 processor
preprocessor, postprocessor = make_carp_pre_post_processors(config, dataset_stats)

print("\n【测试 1: 训练数据 - 包含 task_index】")
print("-" * 80)

# 模拟训练数据 (来自 LIBERO 数据集)
train_sample = {
    "observation.state": torch.randn(8),
    "observation.images.top": torch.randn(3, 480, 640),
    "action": torch.randn(7),
    "task_index": torch.tensor([15]),  # ← 数据集提供的字段
}

print("输入样本:")
print(f"  observation.state: {train_sample['observation.state'].shape}")
print(f"  observation.images.top: {train_sample['observation.images.top'].shape}")
print(f"  action: {train_sample['action'].shape}")
print(f"  task_index: {train_sample['task_index']}")

# 应用 preprocessor
try:
    processed = preprocessor(train_sample)
    print("\n处理后的 batch:")
    for key in sorted(processed.keys()):
        if isinstance(processed[key], torch.Tensor):
            print(f"  {key}: {processed[key].shape}")
        else:
            print(f"  {key}: {processed[key]}")

    # 检查 task_id 是否存在
    if "task_id" in processed:
        print(f"\n✓ 成功: task_id 存在, 值为 {processed['task_id']}")
    else:
        print("\n✗ 失败: task_id 不存在")
        print(f"  可用字段: {list(processed.keys())}")

    # 检查 task_index 是否被移除
    if "task_index" in processed:
        print(f"⚠️  警告: task_index 仍然存在 (应该被重命名)")
    else:
        print(f"✓ task_index 已被正确重命名")

except Exception as e:
    print(f"\n✗ 处理失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("【测试 2: 评估数据 - 环境提供 task_id】")
print("-" * 80)

# 模拟评估数据 (来自 LIBERO 环境)
eval_sample = {
    "observation.state": torch.randn(8),
    "observation.images.top": torch.randn(3, 480, 640),
    "task_id": torch.tensor([5]),  # ← 环境直接提供 task_id
}

print("输入样本:")
print(f"  observation.state: {eval_sample['observation.state'].shape}")
print(f"  observation.images.top: {eval_sample['observation.images.top'].shape}")
print(f"  task_id: {eval_sample['task_id']}")

# 应用 preprocessor
try:
    processed = preprocessor(eval_sample)
    print("\n处理后的 batch:")
    for key in sorted(processed.keys()):
        if isinstance(processed[key], torch.Tensor):
            print(f"  {key}: {processed[key].shape}")
        else:
            print(f"  {key}: {processed[key]}")

    # 检查 task_id 是否保留
    if "task_id" in processed:
        print(f"\n✓ 成功: task_id 保留, 值为 {processed['task_id']}")
    else:
        print("\n✗ 失败: task_id 丢失")

except Exception as e:
    print(f"\n✗ 处理失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("【测试 3: 没有 task 信息的数据】")
print("-" * 80)

# 模拟没有 task 信息的数据
no_task_sample = {
    "observation.state": torch.randn(8),
    "observation.images.top": torch.randn(3, 480, 640),
}

print("输入样本:")
print(f"  observation.state: {no_task_sample['observation.state'].shape}")
print(f"  observation.images.top: {no_task_sample['observation.images.top'].shape}")
print(f"  (没有 task_index 或 task_id)")

# 应用 preprocessor
try:
    processed = preprocessor(no_task_sample)
    print("\n处理后的 batch:")
    for key in sorted(processed.keys()):
        if isinstance(processed[key], torch.Tensor):
            print(f"  {key}: {processed[key].shape}")
        else:
            print(f"  {key}: {processed[key]}")

    # 这种情况下，模型应该使用默认值 (在 forward() 中处理)
    if "task_id" in processed:
        print(f"\n✓ task_id 存在: {processed['task_id']}")
    else:
        print("\n⚠️  task_id 不存在 (模型会使用默认值 0)")

except Exception as e:
    print(f"\n✗ 处理失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("【结论】")
print("-" * 80)
print("Processor 应该:")
print("  1. 将 task_index 重命名为 task_id (训练时)")
print("  2. 保留 task_id 字段 (评估时)")
print("  3. 允许没有 task 信息的样本 (使用默认值)")
print("=" * 80)
