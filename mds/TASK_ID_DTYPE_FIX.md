# Task ID 类型转换错误修复

## 问题描述

运行 `lerobot-eval` 时遇到错误：

```
TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not numpy.ndarray
```

**错误位置**: `CFAP/autoreg.py` line 161
```python
this_ntasks = self.task_embed(ntasks)  # ntasks 是 numpy.ndarray，但期望 Tensor
```

## 原因分析

### 数据流追踪

```
1. add_envs_task() (envs/utils.py)
   ↓
   observation["task_id"] = np.array([15], dtype=np.int64)  ← numpy.ndarray

2. preprocess_observation() (envs/utils.py) - 旧代码
   ↓
   ❌ 没有处理 "task_id" 字段！
   只处理: pixels, robot_state, agent_pos, environment_state
   ↓
   observation["task_id"] 仍然是 numpy.ndarray

3. env_preprocessor(observation)
   ↓
   仍然是 numpy.ndarray

4. preprocessor(observation)
   ↓
   CARPTaskIDProcessorStep 重命名字段但不改变类型
   ↓
   batch["task_id"] 仍然是 numpy.ndarray

5. policy.select_action(batch)
   ↓
   predict_action_chunk()
   ↓
   ar_model.autoregressive_infer_cfg(ntasks=batch["task_id"])
   ↓
   task_embed(ntasks)  ← 需要 Tensor，但收到 numpy.ndarray
   ↓
   ❌ TypeError!
```

### 根本原因

`preprocess_observation()` 函数负责将环境观测从 numpy 转换为 torch.Tensor，但它**没有处理** `task_id` 字段。

旧代码只处理这些字段：
- `pixels` → 转换为 torch.Tensor (uint8 → float32)
- `robot_state` → 递归转换嵌套字典
- `agent_pos` → 转换为 torch.Tensor (float)
- `environment_state` → 转换为 torch.Tensor (float)
- `policy`, `camera_obs` (IsaacLab) → 直接传递

**缺失**: `task_id` 字段

## 解决方案

### 修改的文件

**文件**: `src/lerobot/envs/utils.py`

**修改位置**: `preprocess_observation()` 函数 (line ~108-110)

**添加的代码**:
```python
# Handle task_id (for multi-task policies like CARP)
if "task_id" in observations:
    task_id = observations["task_id"]
    if isinstance(task_id, np.ndarray):
        task_id = torch.from_numpy(task_id).long()
    elif not isinstance(task_id, torch.Tensor):
        task_id = torch.tensor(task_id, dtype=torch.long)
    return_observations["task_id"] = task_id
```

**说明**:
- 检查观测中是否有 `task_id` 字段
- 如果是 numpy array，使用 `torch.from_numpy()` 转换
- 如果是其他类型（list, int），使用 `torch.tensor()` 转换
- 确保 dtype 是 `torch.long` (int64)，因为 `nn.Embedding` 需要整数索引

### 修复后的数据流

```
1. add_envs_task() (envs/utils.py)
   ↓
   observation["task_id"] = np.array([15], dtype=np.int64)

2. preprocess_observation() (envs/utils.py) - 新代码
   ↓
   ✓ 检测到 "task_id" 字段
   ✓ 转换: np.array([15]) → torch.tensor([15], dtype=torch.long)
   ↓
   observation["task_id"] 是 torch.Tensor

3. env_preprocessor(observation)
   ↓
   torch.Tensor (保持不变)

4. preprocessor(observation)
   ↓
   CARPTaskIDProcessorStep 重命名字段
   ↓
   batch["task_id"] 是 torch.Tensor

5. policy.select_action(batch)
   ↓
   predict_action_chunk()
   ↓
   ar_model.autoregressive_infer_cfg(ntasks=batch["task_id"])
   ↓
   task_embed(ntasks)  ← 收到 torch.Tensor
   ↓
   ✓ 成功！
```

## 验证测试

### 测试脚本

`test_task_id_conversion.py`:
```python
mock_observation = {
    "task_id": np.array([15], dtype=np.int64),  # numpy array
    ...
}

processed = preprocess_observation(mock_observation)

# 验证转换
assert isinstance(processed["task_id"], torch.Tensor)
assert processed["task_id"].dtype == torch.int64
```

### 测试结果

```
✓ 类型正确: torch.Tensor
✓ dtype 正确: torch.int64
✓ task_embed 调用成功
```

## 影响范围

### 对 CARP 的影响

- ✅ 修复后，task_id 可以正确传递给 AR 模型
- ✅ task_embed 可以正常工作
- ✅ 多任务条件化正确

### 对其他策略的影响

- ✅ 向后兼容：其他策略不使用 `task_id`，不受影响
- ✅ 可扩展：未来其他多任务策略可以使用相同的机制

### 对其他环境的影响

- ✅ 只有提供 `task_id` 的环境会触发这段代码
- ✅ 不提供 `task_id` 的环境（大多数环境）不受影响

## 相关修复

这是 CARP 评估问题修复系列的第 **4** 个修复：

1. ✅ Task ID 映射（task text → task_index）
2. ✅ ProcessorStep 注册（carp_task_id 等）
3. ✅ RenameObservationsProcessor 缺失
4. ✅ Task ID 类型转换（numpy → tensor）**← 本修复**

## 使用方法

现在可以正常运行评估：

```bash
lerobot-eval \
  --policy.path=/path/to/carp/checkpoint \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=50
```

**预期行为**:
- ✅ LIBERO 环境提供 task description
- ✅ 通过映射获取正确的 dataset task_index (0-39)
- ✅ task_index 作为 numpy array 添加到观测
- ✅ `preprocess_observation()` 转换为 torch.Tensor
- ✅ 传递给 CARP 模型的 task_embed
- ✅ 多任务条件化正常工作

## 总结

| 组件 | 旧行为 | 新行为 |
|------|--------|--------|
| add_envs_task() | 添加 numpy task_id | 添加 numpy task_id ✓ |
| preprocess_observation() | ❌ 忽略 task_id | ✅ 转换为 Tensor |
| preprocessor | task_id 仍是 numpy | task_id 是 Tensor ✓ |
| policy.select_action() | ❌ TypeError | ✅ 成功 |
| task_embed() | ❌ 收到 numpy | ✅ 收到 Tensor |

**关键点**:
- `preprocess_observation()` 是 numpy → tensor 转换的关键点
- 所有环境观测都应该在这里转换为 tensor
- `task_id` 现在遵循相同的模式

---

**修复时间**: 2026-03-23
**修复文件**: `src/lerobot/envs/utils.py` (line ~108-118)
**测试文件**: `test_task_id_conversion.py`
