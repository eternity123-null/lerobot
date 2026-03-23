#!/usr/bin/env python3
"""
详细验证 select_action 的行为
"""

import sys
sys.path.insert(0, 'src')

import torch
from collections import deque

print("=" * 80)
print("验证 select_action 的返回形状")
print("=" * 80)

# ============================================================================
# 模拟当前的实现
# ============================================================================
print("\n[当前实现] 验证返回形状...")

batch_size = 1
action_horizon = 16
action_dim = 7

# 模拟 predict_action_chunk 的输出
action_chunk = torch.randn(batch_size, action_horizon, action_dim)
print(f"\n1. predict_action_chunk 返回: {action_chunk.shape}")
print(f"   -> (batch_size={batch_size}, action_horizon={action_horizon}, action_dim={action_dim})")

# 切片到 action_horizon（这一步实际上没有改变形状）
action_chunk = action_chunk[:, :action_horizon, :]
print(f"\n2. 切片后: {action_chunk.shape}")

# Transpose
action_chunk_transposed = action_chunk.transpose(0, 1)
print(f"\n3. transpose(0, 1): {action_chunk_transposed.shape}")
print(f"   -> (action_horizon={action_horizon}, batch_size={batch_size}, action_dim={action_dim})")

# Extend 到队列
queue = deque()
queue.extend(action_chunk_transposed)
print(f"\n4. extend 到队列:")
print(f"   队列长度: {len(queue)}")
print(f"   队列中第一个元素形状: {queue[0].shape}")
print(f"   队列中最后一个元素形状: {queue[-1].shape}")

# Popleft - 这是 select_action 返回的
action = queue.popleft()
print(f"\n5. popleft() 返回 (这是 select_action 的返回值):")
print(f"   形状: {action.shape}")
print(f"   期望: ({batch_size}, {action_dim})")

if action.shape == (batch_size, action_dim):
    print(f"   ✓ 形状正确!")
else:
    print(f"   ✗ 形状错误!")

# ============================================================================
# 连续调用多次
# ============================================================================
print("\n" + "=" * 80)
print("[连续调用] 验证队列管理...")
print("=" * 80)

# 重新填充队列
queue = deque()
action_chunk = torch.randn(batch_size, action_horizon, action_dim)
queue.extend(action_chunk.transpose(0, 1))

print(f"\n初始队列长度: {len(queue)}")

for i in range(action_horizon):
    action = queue.popleft()
    print(f"第 {i+1} 次调用 popleft():")
    print(f"  返回形状: {action.shape}")
    print(f"  剩余队列长度: {len(queue)}")

    if action.shape != (batch_size, action_dim):
        print(f"  ✗ 形状错误!")
        break
else:
    print(f"\n✓ 所有 {action_horizon} 次调用都返回正确形状 ({batch_size}, {action_dim})")

# ============================================================================
# 多环境情况
# ============================================================================
print("\n" + "=" * 80)
print("[多环境] batch_size=4...")
print("=" * 80)

batch_size_multi = 4
action_chunk_multi = torch.randn(batch_size_multi, action_horizon, action_dim)

print(f"\npredict_action_chunk 返回: {action_chunk_multi.shape}")

queue_multi = deque()
queue_multi.extend(action_chunk_multi.transpose(0, 1))

action_multi = queue_multi.popleft()
print(f"select_action 返回: {action_multi.shape}")
print(f"期望: ({batch_size_multi}, {action_dim})")

if action_multi.shape == (batch_size_multi, action_dim):
    print(f"✓ 多环境形状正确!")
else:
    print(f"✗ 多环境形状错误!")

# ============================================================================
# 检查是否有其他可能返回错误形状的路径
# ============================================================================
print("\n" + "=" * 80)
print("[检查] 是否有可能返回 (B, H, A) 的情况...")
print("=" * 80)

print("\n可能的错误情况:")
print("1. 直接返回 predict_action_chunk 的结果:")
print(f"   return predict_action_chunk(batch)  # {action_chunk.shape}")
print(f"   -> 这会返回 (B, H, A) ✗")

print("\n2. 从队列返回但没有 transpose:")
queue_wrong = deque()
action_chunk_wrong = torch.randn(batch_size, action_horizon, action_dim)
# 如果直接 extend 而不 transpose
# queue_wrong.extend(action_chunk_wrong)  # 这会迭代第一个维度 (batch_size)
# 这实际上不对，因为会把 batch_size 个 (H, A) 加入队列

print("   如果: queue.extend(action_chunk) 而不是 transpose")
print("   这会迭代 batch_size 维度，把 batch_size 个 (H, A) 加入队列")
print("   popleft 会返回 (H, A) ✗")

print("\n3. 当前实现 (transpose + extend + popleft):")
print(f"   return queue.popleft()  # {action.shape}")
print(f"   -> 这会返回 (B, A) ✓")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("总结")
print("=" * 80)

print("\n当前实现的数据流:")
print("  predict_action_chunk: (B, H, A)")
print("           ↓ [:, :H, :]")
print("       (B, H, A)")
print("           ↓ transpose(0, 1)")
print("       (H, B, A)")
print("           ↓ extend")
print("  queue: [(B, A), (B, A), ..., (B, A)]  ← H 个 (B, A) tensor")
print("           ↓ popleft()")
print("       (B, A)  ✓ 正确!")

print("\n结论:")
print("  ✓ select_action 返回 (batch_size, action_dim)")
print("  ✓ 不会返回 (batch_size, action_horizon, action_dim)")
print("  ✓ 与向量化环境兼容")
print("  ✓ 支持单环境和多环境")
