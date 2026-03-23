# 为什么旧代码训练的 Checkpoint 用新代码 Eval 会报错

## 完整流程图解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           训练阶段（旧代码）                                 │
└─────────────────────────────────────────────────────────────────────────────┘

1. 创建 Processor Pipeline
   ┌──────────────────────────────────────────┐
   │ processor_carp.py (旧代码)               │
   │                                          │
   │ if rename_map:  # rename_map = {}       │
   │     ❌ 跳过！                            │
   │     input_steps.append(                  │
   │         RenameObservationsProcessorStep()│
   │     )                                    │
   └──────────────────────────────────────────┘
              ↓
   实际创建的 steps:
   [CARPTaskIDProcessorStep,
    AddBatchDimensionProcessorStep,      ← 没有 RenameObservationsProcessorStep
    NormalizerProcessorStep,
    ...]

2. 保存到磁盘
   ┌──────────────────────────────────────────┐
   │ checkpoint/policy_preprocessor.json      │
   │                                          │
   │ {                                        │
   │   "steps": [                             │
   │     {"registry_name": "carp_task_id"},   │
   │     {"registry_name": "to_batch_..."},   │
   │     {"registry_name": "normalizer_..."},│  ← 没有 rename_observations_processor
   │     ...                                  │
   │   ]                                      │
   │ }                                        │
   └──────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           评估阶段（新代码）                                 │
└─────────────────────────────────────────────────────────────────────────────┘

1. 评估脚本准备 overrides
   ┌──────────────────────────────────────────────┐
   │ lerobot_eval.py (line 537-540)               │
   │                                              │
   │ preprocessor_overrides = {                   │
   │     "device_processor": {...},               │
   │     "rename_observations_processor": {...}, ← 总是传递这个！
   │ }                                            │
   └──────────────────────────────────────────────┘

2. 从 pretrained 加载
   ┌─────────────────────────────────────────────────────┐
   │ PolicyProcessorPipeline.from_pretrained()           │
   │                                                     │
   │ Step 1: 加载 JSON 配置                             │
   │   读取: checkpoint/policy_preprocessor.json        │
   │   ↓                                                 │
   │   loaded_config["steps"] = [                       │
   │     {"registry_name": "carp_task_id"},             │
   │     {"registry_name": "to_batch_..."},             │
   │     ...                                             │
   │   ]                                                 │
   │                                                     │
   │ Step 2: 构建 steps 并应用 overrides                │
   │   遍历 loaded_config["steps"]                      │
   │   如果 step 的 registry_name 在 overrides 中:      │
   │     ✅ 应用 override                                │
   │     ✅ 标记 override 为已使用                       │
   │                                                     │
   │ Step 3: 验证所有 overrides 都被使用 (line 578)    │
   │   ┌────────────────────────────────────────────┐  │
   │   │ _validate_overrides_used()                 │  │
   │   │                                            │  │
   │   │ remaining_overrides = {                    │  │
   │   │     "rename_observations_processor": {...}│  │ ← 没有被使用！
   │   │ }                                          │  │
   │   │                                            │  │
   │   │ available_keys = [                         │  │
   │   │     "carp_task_id",                        │  │
   │   │     "to_batch_processor",                  │  │
   │   │     "normalizer_processor",                │  │
   │   │     ...                                    │  │
   │   │ ]                                          │  │ ← 列表中没有它！
   │   │                                            │  │
   │   │ ❌ raise KeyError(                         │  │
   │   │     "Override keys [...] do not match     │  │
   │   │      any step in the saved configuration" │  │
   │   │ )                                          │  │
   │   └────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────┘
```

## 为什么会有这个验证？

### 设计目的

这个验证机制是**有意设计的安全检查**，目的是：

1. **防止拼写错误**：
   ```python
   # 如果你写错了 step 名称
   overrides = {"normalizer_processer": {...}}  # 拼写错误: processer
   # 验证会报错，而不是静默忽略
   ```

2. **防止配置不匹配**：
   ```python
   # 如果你用 ACT 的 override 去加载 CARP 的 processor
   overrides = {"act_specific_step": {...}}
   # 验证会报错，告诉你这个 step 不存在
   ```

3. **确保 override 生效**：
   ```python
   # 如果你想 override 一个 step，但它根本不在 pipeline 中
   # 验证会告诉你，而不是让你以为 override 生效了
   ```

### 验证代码 (pipeline.py line 1020-1038)

```python
@classmethod
def _validate_overrides_used(
    cls,
    remaining_override_keys: set[str],
    loaded_config: dict[str, Any],
) -> None:
    """验证所有 override keys 都被使用了"""
    if not remaining_override_keys:
        return  # 所有 overrides 都被使用，验证通过

    # 获取保存的配置中有哪些 steps
    available_keys = [
        step.get("registry_name") or step["class"].rsplit(".", 1)[1]
        for step in loaded_config["steps"]
    ]

    # 如果有 override key 没有被使用，报错
    raise KeyError(
        f"Override keys {list(remaining_override_keys)} do not match any step "
        f"in the saved configuration. Available step keys: {available_keys}. "
        f"Make sure override keys match exact step class names or registry names."
    )
```

## 为什么旧 Checkpoint 会报错

### 根本原因

**训练和评估的代码版本不一致**：

| 方面 | 旧代码训练 | 新代码评估 |
|------|-----------|----------|
| processor_carp.py | 跳过 RenameObservationsProcessorStep | 始终包含它 |
| 保存的 JSON | 没有 rename_observations_processor | - |
| lerobot_eval.py | - | 总是 override rename_observations_processor |

**冲突**：
- 评估脚本想要 override 一个 step: `rename_observations_processor`
- 但这个 step 在训练时保存的配置中**根本不存在**
- 验证失败 → KeyError

### 类比解释

想象你有一个配置文件（旧代码训练时保存的）：
```json
{
  "settings": {
    "volume": 50,
    "brightness": 80
  }
}
```

现在你想要覆盖一个设置（新代码评估时）：
```python
overrides = {
    "volume": 100,
    "contrast": 90  # ← 这个设置在配置文件中不存在！
}
```

验证逻辑会说：
> "你想要修改 'contrast'，但配置文件中只有 'volume' 和 'brightness'，没有 'contrast'！"

## 为什么新代码训练就没问题

### 新代码训练

```python
# processor_carp.py (新代码)
# 始终包含，不管 rename_map 是否为空
input_steps.append(RenameObservationsProcessorStep(rename_map={}))
```

**保存的 JSON**：
```json
{
  "steps": [
    {"registry_name": "carp_task_id"},
    {"registry_name": "rename_observations_processor"},  ← 现在有了！
    {"registry_name": "to_batch_processor"},
    ...
  ]
}
```

**评估时**：
```python
preprocessor_overrides = {
    "rename_observations_processor": {...}  ← 在保存的配置中存在
}
# ✅ 验证通过！
```

## 解决方案对比

### 方案 1: 重新训练（推荐）✅

**优点**：
- 彻底解决问题
- 确保训练和评估代码完全一致
- 未来不会有兼容性问题

**缺点**：
- 需要时间重新训练

### 方案 2: 修改评估脚本（不推荐）❌

修改 `lerobot_eval.py`，只在 rename_map 不为空时才 override：

```python
preprocessor_overrides = {
    "device_processor": {"device": str(policy.config.device)},
}
if cfg.rename_map:  # 只有非空时才 override
    preprocessor_overrides["rename_observations_processor"] = {"rename_map": cfg.rename_map}
```

**缺点**：
- 影响所有策略，不只是 CARP
- 可能破坏其他策略的行为
- 不符合 LeRobot 框架的设计意图

### 方案 3: 手动修改 JSON（临时方案）⚠️

手动编辑 `checkpoint/policy_preprocessor.json`，添加：

```json
{
  "steps": [
    {"registry_name": "carp_task_id", ...},
    {
      "step_class": "RenameObservationsProcessorStep",
      "registry_name": "rename_observations_processor",
      "rename_map": {}
    },
    ...
  ]
}
```

**缺点**：
- 容易出错
- 需要知道正确的 JSON 格式
- 不是长期解决方案

## 总结

| 问题 | 为什么会报错 |
|------|------------|
| 训练时（旧代码）| 因为 rename_map 为空，跳过了这个 step |
| 保存的 JSON | 没有包含 rename_observations_processor |
| 评估时（新代码）| lerobot_eval.py 总是尝试 override 它 |
| 验证机制 | 发现 override key 不在保存的配置中 |
| 结果 | KeyError: "rename_observations_processor not found" |

**关键点**：
- ✅ 新代码训练 + 新代码评估 = 没问题
- ❌ 旧代码训练 + 新代码评估 = 报错
- ✅ 旧代码训练 + 旧代码评估 = 没问题（如果有旧代码的话）

**最佳实践**：
1. 保持训练和评估代码版本一致
2. 使用新代码重新训练
3. 确保 processor pipeline 在训练时包含所有可能需要的 steps

---

**文档更新**: 2026-03-23
**核心原因**: Processor 配置序列化/反序列化 + override 验证机制
