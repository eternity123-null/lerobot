# Select Action 形状修复总结

## 问题回顾

运行 `lerobot-eval` 时遇到错误：
```
ValueError: zip() argument 2 is shorter than argument 1
```

## 根本原因

`CARP.select_action()` 返回的动作形状不符合向量化环境的期望：
- **期望**：`(batch_size, action_dim)` - 例如 `(1, 7)` 表示 1 个环境，7 维动作
- **实际**：`(action_dim,)` - 例如 `(7,)` - 缺少 batch 维度

这导致 `gymnasium.vector.SyncVectorEnv.step()` 中的 `zip(actions, envs)` 将 `(7,)` 误认为 7 个环境的动作，而实际只有 1 个环境，从而报错。

## 修复内容

### 修改文件
`src/lerobot/policies/carp/modeling_carp.py` - `select_action()` 方法 (line 348-374)

### 关键变化

**旧实现** (错误):
```python
def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
    if len(self._action_queue) == 0:
        action_chunk = self.predict_action_chunk(batch)  # (B, horizon, action_dim)
        action_chunk = action_chunk.squeeze(0)  # (horizon, action_dim) ← 移除了 batch 维度

        # 迭代 horizon 维度
        for action in action_chunk:
            self._action_queue.append(action)  # action 是 (action_dim,)

    return self._action_queue.popleft()  # 返回 (action_dim,) ← 缺少 batch 维度
```

**新实现** (正确):
```python
def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
    if len(self._action_queue) == 0:
        action_chunk = self.predict_action_chunk(batch)  # (B, horizon, action_dim)
        action_chunk = action_chunk[:, :self.config.action_horizon, :]  # (B, action_horizon, action_dim)

        # Transpose to (action_horizon, B, action_dim) to preserve batch dimension
        self._action_queue.extend(action_chunk.transpose(0, 1))

    return self._action_queue.popleft()  # 返回 (B, action_dim) ← 保留 batch 维度
```

**核心区别**：
- 移除 `squeeze(0)` 和 `for loop + append()`
- 使用 `transpose(0, 1)` + `extend()`
- 队列中存储 `(batch_size, action_dim)` 而不是 `(action_dim,)`

## 形状变换对比

```
旧实现:
  predict_action_chunk: (1, 16, 7)
           ↓ squeeze(0)
       (16, 7)
           ↓ for loop + append
  queue: [(7,), (7,), ..., (7,)]  ← 16 个 (7,) 的 tensor
           ↓ popleft()
        (7,)  ← 缺少 batch 维度 ✗

新实现:
  predict_action_chunk: (1, 16, 7)
           ↓ transpose(0, 1)
       (16, 1, 7)
           ↓ extend
  queue: [(1, 7), (1, 7), ..., (1, 7)]  ← 16 个 (1, 7) 的 tensor
           ↓ popleft()
       (1, 7)  ← 保留 batch 维度 ✓
```

## 多环境支持

修复后同样支持多环境评估：

```python
# 4 个并行环境
batch_size = 4
action_chunk = predict_action_chunk(batch)  # (4, 16, 7)
action_chunk_transposed = action_chunk.transpose(0, 1)  # (16, 4, 7)

# 队列中存储 16 个 (4, 7) 的 tensor
queue.extend(action_chunk_transposed)

# 每次返回 (4, 7) - 4 个环境各自的动作
action = queue.popleft()  # (4, 7)
```

## 与向量化环境的兼容性

```python
# gymnasium.vector.SyncVectorEnv.step() 的期望
actions = policy.select_action(batch)  # (num_envs, action_dim)
actions_numpy = actions.cpu().numpy()

for i, (action, env) in enumerate(zip(actions_numpy, self.envs, strict=True)):
    # action shape: (action_dim,)
    # 每次迭代处理一个环境
    env.step(action)
```

**修复前**：
- `actions` shape: `(7,)` - 被误认为 7 个环境
- `self.envs` 长度: 1
- `zip` 尝试迭代 7 次 vs 1 个环境 → 错误！

**修复后**：
- `actions` shape: `(1, 7)` - 正确表示 1 个环境
- `self.envs` 长度: 1
- `zip` 迭代 1 次 → 成功！

## 与其他策略的一致性

这个修复使 CARP 与 LeRobot 框架中的其他策略保持一致：

| 策略 | select_action 返回形状 | 使用 transpose? |
|------|------------------------|----------------|
| ACT | `(batch_size, action_dim)` | ✓ |
| Diffusion | `(batch_size, action_dim)` | ✓ |
| VQ-BeT | `(batch_size, action_dim)` | ✓ |
| CARP (修复前) | `(action_dim,)` | ✗ |
| CARP (修复后) | `(batch_size, action_dim)` | ✓ |

## 测试验证

### 测试 1: 形状转换逻辑
运行 `test_shape_transformation.py`:
- ✅ 单环境返回 `(1, 7)`
- ✅ 多环境返回 `(4, 7)`
- ✅ 与 zip 操作兼容

### 测试 2: 端到端评估（需要用户验证）
```bash
lerobot-eval \
  --policy.path=/path/to/carp/checkpoint \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=50
```

**预期结果**：
- ✅ 不报 `ValueError: zip() argument 2 is shorter than argument 1`
- ✅ 评估正常运行
- ✅ 环境交互正常

## 影响范围

| 组件 | 是否受影响 | 说明 |
|------|-----------|------|
| 训练 | ✗ | 训练不调用 `select_action()` |
| 评估 | ✓ | 主要修复目标 |
| 实时控制 | ✓ | 机器人控制也使用 `select_action()` |
| 后处理器 | ✓ | 需要处理 `(batch_size, action_dim)` 形状 |
| 其他策略 | ✗ | 不受影响 |

## 向后兼容性

**重要**：这个修复可能影响已有的使用方式（如果有）：

- ✅ **向量化环境评估**：修复后正常工作（之前报错）
- ✅ **标准 lerobot-eval**：修复后正常工作（之前报错）
- ⚠️ **自定义脚本**：如果有脚本假设返回 `(action_dim,)` 形状，需要更新

## 相关文档

- **问题分析**：`ERROR_ANALYSIS_ZIP_ERROR.md`
- **测试脚本**：`test_shape_transformation.py`
- **参考实现**：`src/lerobot/policies/act/modeling_act.py` (line 100-122)

## 总结

| 方面 | 修复前 | 修复后 |
|------|--------|--------|
| 返回形状 | `(action_dim,)` | `(batch_size, action_dim)` |
| 队列操作 | `squeeze + for + append` | `transpose + extend` |
| 向量化环境兼容 | ✗ | ✓ |
| 与框架一致性 | ✗ | ✓ |
| 单环境支持 | ✓ (但报错) | ✓ |
| 多环境支持 | ✗ | ✓ |

**关键点**：
- 这是一个关键的 bug 修复，阻塞了评估功能
- 修复后与 ACT、Diffusion 等策略行为一致
- 遵循 LeRobot 框架的标准模式
- 支持单环境和多环境评估

---

**修复时间**：2026-03-23
**修复文件**：`src/lerobot/policies/carp/modeling_carp.py` (line 348-376)
**严重程度**：关键（阻塞评估）
**修复难度**：简单（单个方法修改）
