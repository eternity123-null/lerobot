# Task ID 注入完整验证报告

## 概述

本报告验证了 CARP 模型在训练和 LIBERO 环境评估时，`task_id` 字段的完整数据流。

## 问题背景

CARP 是一个多任务策略，需要 `task_id` 进行条件化生成：
- **训练时**：数据集提供 `task_index` 字段（0-39 for LIBERO）
- **评估时**：环境需要提供 `task_id` 字段
- **模型期望**：接收 `task_id` 字段进行任务嵌入

然而，LeRobot 框架的命名约定和数据流中存在不一致：
1. 数据集使用 `task_index`，模型期望 `task_id`
2. 评估脚本没有自动将环境的 `task_id` 添加到观测中

## 解决方案

### 1. 创建 CARPTaskIDProcessorStep

**文件**: `src/lerobot/policies/carp/carp_task_id_step.py`

```python
@ProcessorStepRegistry.register(name="carp_task_id")
@dataclass
class CARPTaskIDProcessorStep(ProcessorStep):
    """
    Handle task ID for CARP multi-task policy.

    Renames 'task_index' (from dataset) to 'task_id' (expected by model).
    Preserves existing 'task_id' if already present.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        comp_data = transition.get("complementary_data", {})

        # Rename task_index to task_id in complementary_data
        if "task_index" in comp_data and comp_data["task_index"] is not None:
            comp_data["task_id"] = comp_data.pop("task_index")

        # Handle top-level task_index (from environment)
        if "task_index" in transition and transition["task_index"] is not None:
            comp_data["task_id"] = transition.pop("task_index")

        # Preserve existing task_id
        if "task_id" in transition and transition["task_id"] is not None:
            comp_data["task_id"] = transition.pop("task_id")

        if comp_data:
            transition["complementary_data"] = comp_data

        return transition
```

**关键功能**:
- 将 `task_index` 重命名为 `task_id`
- 处理顶层和 complementary_data 中的字段
- 保留已存在的 `task_id`

### 2. 修改 CARP Preprocessor

**文件**: `src/lerobot/policies/carp/processor_carp.py`

在预处理流水线的**第一步**添加 Task ID 处理：

```python
def make_carp_pre_post_processors(...):
    input_steps = []

    # Step 1: Handle task ID (rename task_index → task_id)
    # IMPORTANT: CARP needs task_id for multi-task conditioning
    input_steps.append(CARPTaskIDProcessorStep())

    # Step 2: Rename observations
    # ...
```

### 3. 修改 Batch Converter

**文件**: `src/lerobot/processor/converters.py`

在 `_extract_complementary_data` 函数中添加 `task_id` 提取：

```python
def _extract_complementary_data(batch: dict[str, Any]) -> dict[str, Any]:
    pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
    task_key = {"task": batch["task"]} if "task" in batch else {}
    index_key = {"index": batch["index"]} if "index" in batch else {}
    task_index_key = {"task_index": batch["task_index"]} if "task_index" in batch else {}
    task_id_key = {"task_id": batch["task_id"]} if "task_id" in batch else {}  # ← 添加
    episode_index_key = {"episode_index": batch["episode_index"]} if "episode_index" in batch else {}

    return {**pad_keys, **task_key, **index_key, **task_index_key, **task_id_key, **episode_index_key}
```

### 4. 修改评估脚本的环境处理

**文件**: `src/lerobot/envs/utils.py`

在 `add_envs_task` 函数中添加 `task_id` 提取：

```python
def add_envs_task(env: gym.vector.VectorEnv, observation: RobotObservation) -> RobotObservation:
    """Adds task feature to the observation dict with respect to the first environment attribute."""

    # ... 原有的 task 处理 ...

    # Add task_id if available (for multi-task policies like CARP)
    # LIBERO and other multi-task envs expose task_id as an attribute
    if hasattr(env.envs[0], "task_id"):
        task_id_result = env.call("task_id")

        if isinstance(task_id_result, tuple):
            task_id_result = list(task_id_result)

        if not isinstance(task_id_result, list):
            raise TypeError(f"Expected task_id to return a list, got {type(task_id_result)}")

        # Convert to numpy array for consistency
        import numpy as np
        observation["task_id"] = np.array(task_id_result, dtype=np.int64)

    return observation
```

## 验证测试

### 测试 1: 训练时的 Task ID 注入

**测试文件**: `test_train_task_id.py`

**测试内容**:
1. 加载 LIBERO 数据集（40 任务，1693 episodes）
2. 检查样本中的 `task_index` 字段
3. 应用 Preprocessor，验证 `task_index` → `task_id` 转换
4. 验证批处理中的 task_id

**结果**: ✓ 全部通过
```
样本 0: 原始 task_index: 0, 处理后 task_id: 0 ✓
样本 100: 原始 task_index: 1, 处理后 task_id: 1 ✓
样本 500: 原始 task_index: 2, 处理后 task_id: 2 ✓
```

### 测试 2: 评估时的 Task ID 注入

**测试文件**: `test_task_id_flow.py`

**测试内容**:
1. 模拟 LIBERO 环境行为（reset/step）
2. 验证 reset() 不返回 task_id（需要手动添加）
3. 验证 step() 返回 info['task_id']
4. 测试 CARPTaskIDProcessorStep 的转换逻辑

**结果**: ✓ 全部通过
```
测试场景 1: task_id 在顶层字段
  ✓ task_id 移动到 complementary_data: 15

测试场景 2: task_index 在 complementary_data
  ✓ task_index 重命名为 task_id: 15

测试场景 3: 验证评估脚本需要的操作
  ✓ 数据流程验证完成
```

## 完整数据流

### 训练流程

```
LeRobotDataset
  ↓
batch["task_index"] = 2
  ↓
Preprocessor (batch_to_transition)
  ↓
transition["complementary_data"]["task_index"] = 2
  ↓
CARPTaskIDProcessorStep
  ↓
transition["complementary_data"]["task_id"] = 2  # ← 重命名
  ↓
transition_to_batch
  ↓
batch["task_id"] = 2
  ↓
CARPPolicy.forward(batch)
  ↓
task_ids = batch.get("task_id", torch.zeros(...))  # ← 模型接收
  ↓
self.ar_model(nobs=..., ntasks=task_ids)  # ← 任务嵌入
```

### 评估流程

```
LIBERO Environment
  ↓
env.reset() → (obs, info)  # info 不含 task_id
  ↓
add_envs_task(env, obs)
  ↓
obs["task_id"] = env.task_id  # ← 从环境属性获取
  ↓
env_preprocessor(obs)
  ↓
preprocessor(obs)
  ↓
CARPTaskIDProcessorStep
  ↓
batch["task_id"] = 15
  ↓
CARPPolicy.select_action(batch)
  ↓
task_ids = batch.get("task_id", torch.zeros(...))  # ← 模型接收
  ↓
self.ar_model(nobs=..., ntasks=task_ids)  # ← 任务嵌入
```

## 关键发现

### 1. LIBERO 环境的 task_id 行为

- **reset()**: 返回 `info = {"is_success": False}`，**不含** task_id
- **step()**: 返回 `info = {"task_id": X, "task": "...", ...}`，**包含** task_id
- **环境属性**: `env.task_id` 始终可用

### 2. 评估脚本的处理

在 `lerobot_eval.py` 的 `rollout()` 函数中：
```python
observation, info = env.reset()  # 第一次循环
observation = add_envs_task(env, observation)  # ← 现在会添加 task_id
```

修改后，`add_envs_task()` 会：
1. 检查 `env.envs[0].task_id` 是否存在
2. 调用 `env.call("task_id")` 获取所有环境的 task_id
3. 添加到 `observation["task_id"]`

### 3. Preprocessor 的职责

`CARPTaskIDProcessorStep` 的职责是：
1. **统一命名**：将 `task_index` 重命名为 `task_id`
2. **保留字段**：确保 `task_id` 不被丢弃
3. **移动位置**：将 task_id 移动到 `complementary_data`（符合 LeRobot 约定）

## 兼容性说明

### 对其他策略的影响

修改的文件中，只有 `envs/utils.py` 的 `add_envs_task()` 会影响其他策略：

- **不会破坏现有策略**：只是添加了新的 `task_id` 字段，不影响已有的 `task` 字段
- **向后兼容**：如果环境没有 `task_id` 属性，则不添加该字段
- **对单任务策略无影响**：单任务策略会忽略 `task_id` 字段

### 对 CARP 的必要性

CARP 模型**必须**接收 task_id：
```python
# modeling_carp.py - forward()
task_ids = batch.get("task_id", torch.zeros(len(batch["action"]), dtype=torch.long, device=self.config.device))
logits_BLV = self.ar_model(nobs=obs_dict, x_BLCv_wo_first_l=x_BLCv_wo_first_l, ntasks=task_ids)
```

- 如果没有 task_id，会默认使用 `torch.zeros(...)`，即 task_id=0
- 这会导致模型对所有任务使用相同的任务嵌入，性能严重下降

## 使用说明

### 训练 CARP

```bash
lerobot-train \
  --policy=carp \
  --dataset.repo_id=/path/to/libero \
  --policy.ar_training_mode=true \
  --policy.vae_checkpoint_path=/path/to/vae/checkpoint \
  --batch_size=64 \
  --steps=200000
```

- ✓ 数据集的 `task_index` 会自动转换为 `task_id`
- ✓ 模型会正确接收 40 个不同的 task ID（0-39）

### 评估 CARP

```bash
lerobot-eval \
  --policy.path=/path/to/carp/checkpoint \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=10
```

- ✓ 环境的 `task_id` 会自动添加到观测中
- ✓ Preprocessor 会正确保留 task_id
- ✓ 模型会使用正确的任务嵌入

## 总结

| 组件 | 文件 | 修改内容 | 状态 |
|------|------|---------|------|
| Task ID Processor Step | `policies/carp/carp_task_id_step.py` | 新建 | ✓ 完成 |
| CARP Preprocessor | `policies/carp/processor_carp.py` | 添加 CARPTaskIDProcessorStep | ✓ 完成 |
| Batch Converter | `processor/converters.py` | 提取 task_id 到 complementary_data | ✓ 完成 |
| 评估环境工具 | `envs/utils.py` | 添加 task_id 到观测 | ✓ 完成 |
| 训练测试 | `test_train_task_id.py` | 验证训练时 task_id 注入 | ✓ 通过 |
| 评估测试 | `test_task_id_flow.py` | 验证评估时 task_id 注入 | ✓ 通过 |

**所有修改已完成并通过测试** ✓

CARP 模型现在可以在 LeRobot 框架中正确进行多任务训练和评估。
