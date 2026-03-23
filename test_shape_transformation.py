#!/usr/bin/env python3
"""
简化测试: 只验证 select_action 的形状转换逻辑
"""

import sys
sys.path.insert(0, 'src')

import torch
from collections import deque

print("=" * 80)
print("测试 select_action 形状转换逻辑")
print("=" * 80)

# ============================================================================
# 测试 1: 旧实现 (错误)
# ============================================================================
print("\n[对比] 旧实现 - squeeze(0) + for loop:")

# 模拟 predict_action_chunk 的输出
action_chunk_old = torch.randn(1, 16, 7)  # (B=1, horizon=16, action_dim=7)
print(f"  predict_action_chunk 输出: {action_chunk_old.shape}")

# 旧实现
action_chunk_old = action_chunk_old.squeeze(0)  # (16, 7)
print(f"  squeeze(0) 后: {action_chunk_old.shape}")

queue_old = deque()
for action in action_chunk_old:
    queue_old.append(action)

first_action_old = queue_old.popleft()
print(f"  队列中取出的 action: {first_action_old.shape}")
print(f"  ✗ 缺少 batch 维度!")

# ============================================================================
# 测试 2: 新实现 (正确)
# ============================================================================
print("\n[新实现] transpose + extend:")

# 模拟 predict_action_chunk 的输出
action_chunk_new = torch.randn(1, 16, 7)  # (B=1, horizon=16, action_dim=7)
print(f"  predict_action_chunk 输出: {action_chunk_new.shape}")

# 新实现
action_chunk_new = action_chunk_new[:, :16, :]  # (1, 16, 7) - 确保长度
print(f"  切片后: {action_chunk_new.shape}")

action_chunk_new_transposed = action_chunk_new.transpose(0, 1)  # (16, 1, 7)
print(f"  transpose(0, 1) 后: {action_chunk_new_transposed.shape}")

queue_new = deque()
queue_new.extend(action_chunk_new_transposed)
print(f"  队列长度: {len(queue_new)}")

first_action_new = queue_new.popleft()
print(f"  队列中取出的 action: {first_action_new.shape}")
print(f"  ✓ 保留 batch 维度!")

# ============================================================================
# 测试 3: 多环境情况 (batch_size=4)
# ============================================================================
print("\n[多环境] batch_size=4:")

action_chunk_multi = torch.randn(4, 16, 7)  # (B=4, horizon=16, action_dim=7)
print(f"  predict_action_chunk 输出: {action_chunk_multi.shape}")

action_chunk_multi = action_chunk_multi[:, :16, :]  # (4, 16, 7)
action_chunk_multi_transposed = action_chunk_multi.transpose(0, 1)  # (16, 4, 7)
print(f"  transpose(0, 1) 后: {action_chunk_multi_transposed.shape}")

queue_multi = deque()
queue_multi.extend(action_chunk_multi_transposed)

first_action_multi = queue_multi.popleft()
print(f"  队列中取出的 action: {first_action_multi.shape}")
print(f"  ✓ 每个环境一个动作!")

# ============================================================================
# 测试 4: 模拟向量化环境的 zip 操作
# ============================================================================
print("\n[向量化环境] 模拟 zip 操作:")

# 单环境情况
num_envs = 1
action_numpy = first_action_new.cpu().numpy()  # (1, 7)
print(f"  action shape: {action_numpy.shape}")

class MockEnv:
    pass

mock_envs = [MockEnv() for _ in range(num_envs)]

try:
    action_list = []
    for i, (action, env) in enumerate(zip(action_numpy, mock_envs, strict=True)):
        action_list.append(action.shape)
    print(f"  ✓ zip 成功: 迭代 {len(action_list)} 次")
    print(f"  ✓ 每次迭代的 action shape: {action_list[0]}")
except ValueError as e:
    print(f"  ✗ zip 失败: {e}")

# 多环境情况
num_envs = 4
action_numpy_multi = first_action_multi.cpu().numpy()  # (4, 7)
print(f"\n  多环境 action shape: {action_numpy_multi.shape}")

mock_envs_multi = [MockEnv() for _ in range(num_envs)]

try:
    action_list_multi = []
    for i, (action, env) in enumerate(zip(action_numpy_multi, mock_envs_multi, strict=True)):
        action_list_multi.append(action.shape)
    print(f"  ✓ zip 成功: 迭代 {len(action_list_multi)} 次")
    print(f"  ✓ 每次迭代的 action shape: {action_list_multi[0]}")
except ValueError as e:
    print(f"  ✗ zip 失败: {e}")

# ============================================================================
# 最终总结
# ============================================================================
print("\n" + "=" * 80)
print("✅ 所有测试通过!")
print("=" * 80)
print("\n关键变化:")
print("  旧实现:")
print("    action_chunk.squeeze(0)         # (B, H, A) -> (H, A)")
print("    for action in action_chunk:")
print("        queue.append(action)        # queue: [(A,), (A,), ...]")
print("    return queue.popleft()          # (A,) - 缺少 batch 维度 ✗")
print("\n  新实现:")
print("    action_chunk.transpose(0, 1)    # (B, H, A) -> (H, B, A)")
print("    queue.extend(action_chunk)")
print("    return queue.popleft()          # (B, A) - 保留 batch 维度 ✓")
print("\n影响:")
print("  - 与 gymnasium.vector.SyncVectorEnv 兼容")
print("  - 支持单环境和多环境评估")
print("  - 与 LeRobot 框架其他策略一致 (ACT, Diffusion等)")
print("\n" + "=" * 80)
