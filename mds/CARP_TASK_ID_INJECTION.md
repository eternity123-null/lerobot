# CARP Task ID Injection 机制说明

## 问题概述

CARP 是多任务模型，需要 **task ID** 作为条件输入来生成不同任务的动作。本文档说明了在训练和评估时如何将 task ID 注入到模型中。

---

## 1. 训练时的 Task ID 处理

### 1.1 数据集中的 Task 信息

**LIBERO 数据集包含 `task_index` 字段**:

```json
// /path/to/libero/meta/info.json
{
    "features": {
        "task_index": {
            "dtype": "int64",
            "shape": [1],
            "names": null,
            "fps": 10.0
        }
    }
}
```

- `task_index` 范围: 0 到 39 (LIBERO 有 40 个任务)
- 每个样本都包含对应的任务索引

### 1.2 模型期望的输入格式

**CARPPolicy.forward() 期望**:

```python
def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    """
    Args:
        batch: {
            "observation.state": (B, n_obs_steps, D),
            "observation.images.*": (B, n_obs_steps, C, H, W),
            "action": (B, action_horizon, A),
            "task_id": (B,) - 期望这个字段！
        }
    """
    # 提取 task_id (如果不存在，默认为 0)
    task_ids = batch.get("task_id", torch.zeros(len(batch["action"]), dtype=torch.long))
```

**问题**: 数据集字段名是 `task_index`，模型期望的是 `task_id`

### 1.3 解决方案：添加 Processor Step

**方案 1: 在 RenameObservationsProcessorStep 中添加映射** ✅ **推荐**

修改 `processor_carp.py`:

```python
def make_carp_pre_post_processors(...):
    # Step 1: Rename observations AND task_index
    rename_map = {
        "task_index": "task_id",  # 重命名数据集字段
    }
    if rename_map:
        input_steps.append(RenameObservationsProcessorStep(rename_map=rename_map))
```

**方案 2: 创建专用的 Task ID Processor Step**

如果需要更复杂的处理（例如 task ID 映射、验证等）:

```python
@ProcessorStepRegistry.register(name="carp_task_id")
@dataclass
class CARPTaskIDProcessorStep(ProcessorStep):
    """Extract and rename task_index to task_id for CARP."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if "task_index" in transition["observation"]:
            transition["observation"]["task_id"] = transition["observation"].pop("task_index")
        elif "task_id" not in transition["observation"]:
            # 如果都不存在，默认为 0
            transition["observation"]["task_id"] = torch.zeros(1, dtype=torch.long)
        return transition
```

### 1.4 当前实现状态

**❌ 问题**: 当前 `processor_carp.py` 没有处理 `task_index` → `task_id` 的重命名

**结果**:
- 模型在训练时会使用默认值 `task_id=0`
- 所有任务都被当作同一个任务训练
- **多任务能力未被正确利用**

---

## 2. 评估时的 Task ID 处理

### 2.1 LIBERO 环境提供 Task ID

**LiberoEnv.step() 返回的 info 包含 task_id**:

```python
# src/lerobot/envs/libero.py
class LiberoEnv(gym.Env):
    def __init__(self, ..., task_id: int, ...):
        self.task_id = task_id  # 保存任务 ID

    def step(self, action):
        raw_obs, reward, done, info = self._env.step(action)
        info.update({
            "task": self.task,
            "task_id": self.task_id,  # ← 添加到 info 中
            "done": done,
            "is_success": is_success,
        })
        return observation, reward, terminated, truncated, info
```

### 2.2 Evaluation Loop 如何传递 Task ID

**LeRobot 评估流程** (`scripts/lerobot_eval.py`):

```python
# 1. 环境 reset 时获取初始观测
obs, info = env.reset()

# 2. 循环执行
for step in range(max_steps):
    # 从 info 中提取 task_id (如果存在)
    if "task_id" in info:
        obs["task_id"] = info["task_id"]

    # 预处理观测
    batch = preprocessor(obs)

    # 模型推理
    action = policy.select_action(batch)

    # 环境交互
    obs, reward, terminated, truncated, info = env.step(action)
```

**关键点**:
1. `info` 中的 `task_id` 需要被添加到 `obs` 字典中
2. `preprocessor` 需要保留 `task_id` 字段（不能丢弃）
3. `batch` 最终传递给 `policy.select_action()`

### 2.3 当前实现状态

**✅ 环境侧**: LIBERO 环境正确提供 `task_id`

**❌ 评估脚本侧**: 需要确认评估脚本是否将 `info["task_id"]` 添加到观测中

**❌ Processor 侧**: 需要确认 processor 是否保留 `task_id` 字段

---

## 3. AR 模型中的 Task Embedding

### 3.1 Task Embedding 层

```python
# src/lerobot/policies/carp/CFAP/autoreg.py
class Coarse2FineAutoRegressor(nn.Module):
    def __init__(self, ..., task_num=8, task_embed_dim=3):
        self.task_embed = nn.Embedding(task_num, task_embed_dim)

    def forward(self, nobs, x_BLCv_wo_first_l, ntasks):
        # 1. 编码 task ID
        this_ntasks = self.task_embed(ntasks)  # [B, task_embed_dim]

        # 2. 编码观测
        nobs_features = self.obs_encoder(nobs)  # [B, obs_dim]

        # 3. 拼接 task embedding 和 obs features
        nobs_features = torch.cat((nobs_features, this_ntasks), dim=-1)
        # [B, obs_dim + task_embed_dim]

        # 4. 投影到 Transformer embedding space
        obs_emb = self.obs_embed(nobs_features)  # [B, embed_dim]

        # 5. 作为条件输入到 Transformer
        ...
```

### 3.2 Task ID 的作用

**条件化生成**:
- Task embedding 与观测特征拼接后，作为条件输入
- Transformer 的每一层都通过 Adaptive Layer Normalization (AdaLN) 接收这个条件
- 不同的 task ID → 不同的 task embedding → 不同的动作生成策略

**LIBERO 配置**:
- `task_num=40`: 40 个任务
- `task_embed_dim=3`: 每个任务用 3 维向量表示
- Task Embedding 参数量: 40 × 3 = 120 个参数

---

## 4. 完整数据流

### 4.1 训练时

```
LeRobotDataset
    ↓ 加载样本
{
    "observation.state": (D,),
    "observation.images.image": (H, W, C),
    "action": (A,),
    "task_index": (1,)  ← 数据集字段
}
    ↓ Preprocessor
    ↓ Step 1: RenameObservations
{
    "observation.state": (D,),
    "observation.images.image": (H, W, C),
    "action": (A,),
    "task_id": (1,)  ← 重命名后
}
    ↓ Step 2-6: AddBatch, Normalize, SampleSequence, AddTemporal, Device
{
    "observation.state": (B, n_obs_steps, D),
    "observation.images.image": (B, n_obs_steps, C, H, W),
    "action": (B, action_horizon, A),
    "task_id": (B,)
}
    ↓ CARPPolicy.forward()
task_ids = batch.get("task_id", default=0)
    ↓ AR model
self.task_embed(task_ids) → condition
```

### 4.2 评估时

```
LiberoEnv
    ↓ reset() / step()
obs, info = env.reset()
info["task_id"] = self.task_id  ← 环境提供
    ↓ Evaluation Loop
obs["task_id"] = info["task_id"]  ← 添加到观测
    ↓ Preprocessor (same as training)
{
    "observation.state": (B, n_obs_steps, D),
    "observation.images.image": (B, n_obs_steps, C, H, W),
    "task_id": (B,)
}
    ↓ CARPPolicy.select_action()
task_ids = batch.get("task_id", default=0)
    ↓ AR model (autoregressive inference)
actions = ar_model.autoregressive_infer_cfg(
    nobs=obs_dict,
    ntasks=task_ids,  ← 条件输入
    vae_proxy=vae
)
```

---

## 5. 需要修复的问题

### ❌ 问题 1: Processor 没有重命名 task_index → task_id

**位置**: `src/lerobot/policies/carp/processor_carp.py`

**修复**:
```python
def make_carp_pre_post_processors(...):
    # Step 1: Rename observations AND task_index
    rename_map = {
        "task_index": "task_id",  # ← 添加这一行
    }
    if rename_map:
        input_steps.append(RenameObservationsProcessorStep(rename_map=rename_map))
```

### ❌ 问题 2: 评估时未将 task_id 添加到观测

**位置**: `src/lerobot/scripts/lerobot_eval.py` (或评估逻辑)

**需要确认**: 评估脚本是否执行了 `obs["task_id"] = info["task_id"]`

如果没有，需要添加此逻辑。

### ❌ 问题 3: Processor 可能丢弃 task_id 字段

**需要检查**:
- `AddBatchDimensionProcessorStep` 是否保留非标准字段
- `NormalizerProcessorStep` 是否只处理已知特征
- `DeviceProcessorStep` 是否移动所有张量

**可能需要**: 在 processor 中明确声明 `task_id` 为保留字段

---

## 6. 推荐的修复方案

### Step 1: 修改 processor_carp.py

```python
def make_carp_pre_post_processors(
    config: CARPConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[...]:
    # ========== Preprocessor ==========
    input_steps = []

    # Step 1: Rename task_index to task_id
    rename_map = {
        "task_index": "task_id",  # ← 关键修复
    }
    input_steps.append(RenameObservationsProcessorStep(rename_map=rename_map))

    # ... 其他步骤保持不变
```

### Step 2: 确认评估脚本传递 task_id

检查 `lerobot_eval.py` 或评估工具是否正确传递 `info["task_id"]` 到观测。

LeRobot 的标准评估流程应该已经处理了这个，但需要验证。

### Step 3: 验证训练命令

确保训练命令中指定了正确的 `task_num`:

```bash
lerobot-train \
  --policy.type=carp \
  --policy.task_num=40 \  # ← LIBERO 有 40 个任务
  ...
```

---

## 7. 测试验证

### 7.1 训练时验证

在训练开始后，添加调试代码:

```python
# 在 CARPPolicy.forward() 中
def forward(self, batch):
    task_ids = batch.get("task_id", torch.zeros(...))
    print(f"Task IDs in batch: {task_ids}")  # 应该看到非零值
    ...
```

**预期**: 应该看到 task_ids 在 [0, 39] 范围内变化

### 7.2 评估时验证

在评估循环中添加:

```python
batch = preprocessor(obs)
print(f"Task ID in batch: {batch.get('task_id', 'MISSING')}")
```

**预期**: 应该看到对应任务的 task_id（例如 `libero_spatial` 的某个特定 ID）

---

## 8. 总结

### 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| **数据集** | ✅ 正确 | LIBERO 包含 `task_index` 字段 |
| **环境** | ✅ 正确 | LIBERO 环境提供 `task_id` |
| **模型** | ✅ 正确 | AR 模型正确使用 task embedding |
| **Processor** | ❌ **缺失** | 没有重命名 `task_index` → `task_id` |
| **评估脚本** | ❓ 待确认 | 需要确认 task_id 传递逻辑 |

### 必须修复

1. **修改 `processor_carp.py`**: 添加 `task_index` → `task_id` 重命名
2. **验证评估脚本**: 确认 task_id 正确传递
3. **测试**: 验证训练和评估时 task_id 正确注入

### 影响

**不修复的后果**:
- ✅ 模型仍然可以训练（使用默认 task_id=0）
- ❌ 多任务学习能力丧失（所有任务被当作单一任务）
- ❌ 评估时 task conditioning 失效
- ❌ 性能可能下降（无法利用任务特异性）

**修复后的好处**:
- ✅ 正确的多任务学习
- ✅ 每个任务有独立的 task embedding
- ✅ 更好的泛化能力
- ✅ 符合原始 CARP 设计

---

**文档日期**: 2026-03-22
**状态**: 🔴 需要修复
**优先级**: 高
