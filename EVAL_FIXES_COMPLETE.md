# CARP 评估问题修复完成报告

## 修复的问题

### 问题 1: Task ID 映射不一致 ✓ 已修复

**问题描述**:
- LIBERO 环境的 task_id 是 per-suite 的（每个 suite 内 0-9）
- 数据集的 task_index 是 global 的（0-39），且是乱序排列的
- 不能简单通过 `suite_offset + local_task_id` 计算

**解决方案**:
- 创建硬编码的 task text → task_index 映射字典
- 在评估时，通过环境的 task description 查找对应的 dataset task_index
- 将正确的 task_index 传递给 CARP 模型

**修改的文件**:
1. **新建**: `src/lerobot/envs/libero_task_mapping.py`
   - 包含 40 个任务的 text → index 映射
   - 提供 `get_task_index_from_description()` 辅助函数

2. **修改**: `src/lerobot/envs/utils.py`
   - 函数 `add_envs_task()` (line 186-200)
   - 添加 LIBERO-specific 逻辑：使用 task description 查找 task_index
   - 保持向后兼容：非 LIBERO 环境仍使用原有逻辑

**验证结果**:
```
libero_spatial[0]: env_task_id=0 → dataset_task_index=34 ✓
libero_object[0]:  env_task_id=0 → dataset_task_index=24 ✓
libero_goal[0]:    env_task_id=0 → dataset_task_index=19 ✓
libero_10[0]:      env_task_id=0 → dataset_task_index=5  ✓
```

**数据集乱序证明**:
- libero_spatial 的 10 个任务映射到: [34, 37, 38, 30, 36, 31, 32, 39, 33, 35]
- libero_object 的 10 个任务映射到: [24, 22, 26, 29, 25, 28, 21, 20, 27, 23]
- libero_goal 的 10 个任务映射到: [19, 17, 14, 18, 11, 13, 10, 16, 15, 12]
- libero_10 的 10 个任务映射到: [5, 7, 3, 1, 6, 4, 9, 0, 8, 2]

### 问题 2: ProcessorStep 未注册 ✓ 已修复

**问题描述**:
```
KeyError: "Processor step 'carp_task_id' not found in registry.
Available steps: [...] (carp_task_id 不在列表中)"
```

**原因分析**:
- `policies/carp/__init__.py` 是空文件
- 没有导入 processor step 模块
- `@ProcessorStepRegistry.register()` 装饰器从未执行
- Registry 中没有 CARP steps

**解决方案**:
- 在 `policies/carp/__init__.py` 中导入所有 processor step 类
- 触发装饰器执行，将 steps 注册到 ProcessorStepRegistry
- 遵循 XVLA 和 Groot 的标准模式

**修改的文件**:
- **修改**: `src/lerobot/policies/carp/__init__.py`
  - 从空文件改为导入 3 个 processor step 类
  - 导入 `CARPTaskIDProcessorStep`
  - 导入 `CARPSampleActionSequenceStep`
  - 导入 `CARPAddTemporalDimensionStep`

**验证结果**:
```
Registered steps: 40 (之前是 37)
CARP steps (3):
  - carp_add_temporal_dimension ✓
  - carp_sample_action_sequence ✓
  - carp_task_id ✓
```

## 技术细节

### Task Description → Task Index 映射示例

| Suite | Env Task ID | Dataset Task Index | Task Description |
|-------|-------------|--------------------|------------------|
| libero_spatial | 0 | 34 | "pick up the black bowl between the plate..." |
| libero_spatial | 1 | 37 | "pick up the black bowl next to the ramekin..." |
| libero_object | 0 | 24 | "pick up the alphabet soup and place it..." |
| libero_object | 1 | 22 | "pick up the cream cheese and place it..." |
| libero_goal | 0 | 19 | "open the middle drawer of the cabinet" |
| libero_10 | 0 | 5 | "put both the alphabet soup and the tomato..." |

### 数据流对比

**修复前（错误）**:
```
LIBERO Env
  ↓
env.task_id (per-suite: 0-9)
  ↓
observation["task_id"] = per_suite_task_id  # ← 错误！
  ↓
CARP Model (期望 global task_index: 0-39)
```

**修复后（正确）**:
```
LIBERO Env
  ↓
env.task_description (task text)
  ↓
get_task_index_from_description(task_text)
  ↓
observation["task_id"] = dataset_task_index  # ← 正确！
  ↓
CARP Model (接收到正确的 task_index: 0-39)
```

## 影响范围

### 对 CARP 的影响
- ✅ 修复后，CARP 在评估时接收到与训练时一致的 task_id
- ✅ 多任务学习的条件化信息正确
- ✅ 可以正常运行 `lerobot-eval` 命令

### 对其他策略的影响
- ✅ 向后兼容：非 LIBERO 环境不受影响
- ✅ 不影响单任务策略（它们会忽略 task_id 字段）
- ✅ 为未来其他多任务策略提供了模板

## 文件清单

### 新建文件
1. `src/lerobot/envs/libero_task_mapping.py` (70 lines)
   - 硬编码的 40 任务映射字典
   - 辅助函数 `get_task_index_from_description()`

### 修改文件
1. `src/lerobot/envs/utils.py`
   - 函数 `add_envs_task()`: +35 lines
   - 添加 LIBERO task description → task_index 映射逻辑

2. `src/lerobot/policies/carp/__init__.py`
   - 从空文件改为 17 lines
   - 导入 3 个 processor step 类

### 测试文件
1. `test_task_mapping.py` (新建)
   - 验证 task text → task_index 映射
   - 测试所有 4 个 LIBERO suites

## 使用方法

### 评估 CARP 模型

```bash
# 现在可以正常运行评估
lerobot-eval \
  --policy.path=/path/to/carp/checkpoint \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=50 \
  --eval.batch_size=10
```

**行为变化**:
- **之前**: 环境 task_id=0 → 模型接收 task_id=0（错误）
- **之后**: 环境 task_id=0 → 查找描述 → 模型接收 task_index=34（正确）

### 训练 CARP 模型（不受影响）

```bash
# 训练逻辑不需要修改
lerobot-train \
  --policy=carp \
  --dataset.repo_id=/path/to/libero \
  --policy.ar_training_mode=true \
  --policy.vae_checkpoint_path=/path/to/vae
```

数据集已经提供正确的 task_index (0-39)，训练不受影响。

## 验证测试

### 测试 1: Processor Steps 注册 ✓
```bash
python3 -c "
from lerobot.policies.carp import CARPConfig
from lerobot.processor import ProcessorStepRegistry
print('CARP steps:', [s for s in ProcessorStepRegistry.list() if 'carp' in s])
"
# 输出: ['carp_add_temporal_dimension', 'carp_sample_action_sequence', 'carp_task_id']
```

### 测试 2: Task Text 映射 ✓
```bash
python test_task_mapping.py
# 输出: 所有 40 个任务都成功映射
```

### 测试 3: 端到端评估（待用户运行）

```bash
# 需要用户提供训练好的 checkpoint
lerobot-eval \
  --policy.path=/path/to/trained/carp/checkpoint \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=1 \
  --eval.batch_size=1
```

**预期结果**:
- ✅ 不报 KeyError: "Processor step 'carp_task_id' not found"
- ✅ 模型正确接收 dataset task_index
- ✅ 评估可以正常运行完成

## 潜在问题和注意事项

### 1. 数据集更新
如果 LIBERO 数据集更新，需要同步更新 `libero_task_mapping.py` 中的映射字典。

**解决方案**:
```python
# 可以从数据集自动生成映射
import pandas as pd
tasks_df = pd.read_parquet('/path/to/libero/meta/tasks.parquet')
mapping = {idx: row['task_index'] for idx, row in tasks_df.iterrows()}
```

### 2. Libero_90 Suite
当前映射只包含 40 个任务（libero_spatial, libero_object, libero_goal, libero_10）。
如果评估 libero_90 suite，需要扩展映射字典。

### 3. 性能影响
- Task text 查找是 O(1) 字典操作，性能影响可忽略
- Processor step 注册在启动时一次性完成，运行时无影响

## 下一步工作

1. **完整训练验证**:
   - 训练完整的 VAE (200k steps)
   - 训练完整的 AR (200k steps)
   - 验证 task conditioning 是否有效提升性能

2. **性能基准测试**:
   - 在 libero_spatial (10 tasks) 上评估
   - 在 libero_object (10 tasks) 上评估
   - 对比不同 task_id 的成功率

3. **文档更新**:
   - 更新 INTEGRATION_COMPLETE.md（标记问题已解决）
   - 更新 TASK_ID_INJECTION_COMPLETE.md（添加 mapping 说明）

---

**修复完成时间**: 2026-03-23
**修复人**: Claude Code
**状态**: ✅ 两个问题均已修复并验证通过
