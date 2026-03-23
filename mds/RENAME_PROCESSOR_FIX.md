# CARP 评估 RenameObservationsProcessor 错误修复

## 问题描述

运行 `lerobot-eval` 时遇到错误：

```
KeyError: "Override keys ['rename_observations_processor'] do not match any step
in the saved configuration. Available step keys: ['carp_task_id', 'to_batch_processor',
'normalizer_processor', 'carp_sample_action_sequence', 'carp_add_temporal_dimension',
'device_processor']."
```

## 原因分析

1. **评估脚本的默认行为**：
   - `lerobot_eval.py` (line 537-540) 会为所有策略传递 `rename_observations_processor` 的override
   - 这是为了支持在评估时重新映射观测键名

2. **CARP训练时的问题**：
   - 在修复前，`processor_carp.py` 只在 `rename_map` 不为空时才添加 RenameObservationsProcessorStep
   - 由于默认 `rename_map = {}`，这个step被跳过了
   - 训练时保存的processor配置中没有这个step

3. **加载时的冲突**：
   - 从pretrained加载processor时，验证override keys
   - 发现 `rename_observations_processor` 不在保存的配置中
   - 报KeyError

## 解决方案

### 修改的文件

**文件**: `src/lerobot/policies/carp/processor_carp.py`

**修改前** (line 74-77):
```python
# Step 2: Rename observations (if needed for other fields)
rename_map = {}  # Can be customized based on robot/env naming conventions
if rename_map:
    input_steps.append(RenameObservationsProcessorStep(rename_map=rename_map))
```

**修改后** (line 74-76):
```python
# Step 2: Rename observations (if needed for other fields)
# Always include this step (even with empty rename_map) for eval compatibility
input_steps.append(RenameObservationsProcessorStep(rename_map={}))
```

### 修改原理

1. **始终包含step**：即使 `rename_map` 是空字典，也添加 RenameObservationsProcessorStep
2. **空rename_map的行为**：当rename_map为空时，这个step不做任何转换，直接返回原始观测
3. **与其他策略一致**：ACT, Diffusion等策略都是这样做的（参考 `processor_act.py` line 57）

## 影响范围

### 对新训练的模型

✅ **完全修复**：
- 使用修改后的代码训练的模型会在processor配置中包含 `rename_observations_processor`
- 评估时可以正常加载和override

### 对已有的checkpoint

⚠️ **仍然会报错**：
- 如果checkpoint是用修改前的代码训练的，保存的processor配置中没有这个step
- 从pretrained加载时仍然会报相同的错误

**解决方法（针对已有checkpoint）**：

#### 方法1: 重新训练（推荐）

```bash
# 使用修改后的代码重新训练
lerobot-train --policy=carp --dataset.repo_id=... --steps=200000
```

#### 方法2: 手动修改已保存的processor配置

编辑 checkpoint 目录下的 `policy_preprocessor.json`，在 `steps` 列表中添加：

```json
{
  "steps": [
    {"step_class": "CARPTaskIDProcessorStep", "registry_name": "carp_task_id", ...},
    {
      "step_class": "RenameObservationsProcessorStep",
      "registry_name": "rename_observations_processor",
      "rename_map": {}
    },
    ...其他steps...
  ]
}
```

**注意**：手动修改配置文件容易出错，不推荐。

#### 方法3: 修改评估脚本（临时方案）

在 `lerobot_eval.py` 中，检查processor是否有这个step：

```python
# 修改前 (line 537-540)
preprocessor_overrides = {
    "device_processor": {"device": str(policy.config.device)},
    "rename_observations_processor": {"rename_map": cfg.rename_map},
}

# 修改后
preprocessor_overrides = {
    "device_processor": {"device": str(policy.config.device)},
}
# 只有在rename_map不为空时才override
if cfg.rename_map:
    preprocessor_overrides["rename_observations_processor"] = {"rename_map": cfg.rename_map}
```

但这会影响所有策略，不推荐。

## 验证

### 新训练的模型

1. 训练一个新的CARP模型：
```bash
lerobot-train \
  --policy=carp \
  --dataset.repo_id=/path/to/libero \
  --policy.ar_training_mode=false \
  --batch_size=8 \
  --steps=1000 \
  --output_dir=outputs/carp_test_new
```

2. 检查保存的processor配置：
```bash
cat outputs/carp_test_new/checkpoints/001000/pretrained_model/policy_preprocessor.json | grep -A2 rename_observations
```

应该能看到：
```json
{
  "step_class": "RenameObservationsProcessorStep",
  "registry_name": "rename_observations_processor",
  "rename_map": {}
}
```

3. 测试评估：
```bash
lerobot-eval \
  --policy.path=outputs/carp_test_new/checkpoints/001000/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=1
```

应该不再报 KeyError。

## 总结

| 问题 | 原因 | 修复方法 | 状态 |
|------|------|---------|------|
| KeyError: rename_observations_processor not found | 训练时未包含这个step | 始终包含RenameObservationsProcessorStep | ✅ 已修复 |
| 已有checkpoint仍报错 | checkpoint用旧代码训练 | 重新训练或手动修改配置 | ⚠️ 需要用户操作 |

**建议**：
- ✅ 使用修改后的代码训练新模型
- ✅ 所有未来的训练都会包含这个step
- ⚠️ 已有的checkpoint需要重新训练

---

**修复时间**: 2026-03-23
**修复文件**: `src/lerobot/policies/carp/processor_carp.py` (line 74-76)
**参考**: ACT processor (`processor_act.py` line 57)
