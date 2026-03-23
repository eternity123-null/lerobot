import torch
import sys
sys.path.insert(0, 'src')

from lerobot.policies.carp.carp_task_id_step import CARPTaskIDProcessorStep
from lerobot.processor.core import EnvTransition

print("测试 CARPTaskIDProcessorStep")
print("=" * 60)

step = CARPTaskIDProcessorStep()

# 测试 1: task_index 存在
transition1 = {
    "observation": {"state": torch.randn(8)},
    "action": torch.randn(7),
    "task_index": torch.tensor([15]),
}

print("\n输入:")
print(f"  keys: {list(transition1.keys())}")
if "task_index" in transition1:
    print(f"  task_index: {transition1['task_index']}")

result1 = step(transition1)

print("\n输出:")
print(f"  keys: {list(result1.keys())}")
if "task_id" in result1:
    print(f"  task_id: {result1['task_id']}")
    print("✓ task_index → task_id 重命名成功")
else:
    print("✗ task_id 不存在")
if "task_index" in result1:
    print("✗ task_index 仍然存在")

# 测试 2: task_id 已存在
print("\n" + "=" * 60)
transition2 = {
    "observation": {"state": torch.randn(8)},
    "action": torch.randn(7),
    "task_id": torch.tensor([5]),
}

print("\n输入:")
print(f"  keys: {list(transition2.keys())}")
if "task_id" in transition2:
    print(f"  task_id: {transition2['task_id']}")

result2 = step(transition2)

print("\n输出:")
print(f"  keys: {list(result2.keys())}")
if "task_id" in result2:
    print(f"  task_id: {result2['task_id']}")
    print("✓ task_id 保留成功")
else:
    print("✗ task_id 丢失")
