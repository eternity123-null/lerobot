# Action Shape 问题完整修复

## 问题描述

用户报告：`lerobot-eval` 中拿到的 Action shape 是 `torch.Size([1, 16, 7])`，而不是期望的 `torch.Size([1, 7])`。

## 调试输出分析

```
[select_action] predict_action_chunk returned: torch.Size([1, 1, 16, 7])  ← 4维！
[select_action] after slice: torch.Size([1, 1, 16, 7])
[select_action] after transpose: torch.Size([1, 1, 16, 7])  ← transpose(0,1) 没有效果
[select_action] queue length after extend: 1  ← 只添加了1个元素
[select_action] first queue element shape: torch.Size([1, 16, 7])
[select_action] returning action: torch.Size([1, 16, 7])  ← 错误！
*****Action shape: torch.Size([1, 16, 7]) *****
```

## 根本原因

### VAE 的 `fhat_to_action` 方法返回 4 维 tensor

**文件**: `src/lerobot/policies/carp/MSAT/vqvae.py` line 120

```python
def fhat_to_action(self, f_hat: torch.Tensor):
    """
    @input: f_hat has shape [B, Cvae, last_l, last_w]
    """
    # ... 解码逻辑 ...
    actions = torch.cat(actions, dim=-1)  # -> [B, 1, 16, action_dim]
    return actions  # ← 返回 4 维 tensor!
```

**返回形状**: `[B, 1, action_horizon, action_dim]` = `[1, 1, 16, 7]`

### 为什么会有这个额外的维度？

这是 CARP 原始实现的设计：
- VAE 将动作序列视为"图像": `(B, C, H, W)` 格式
- C=1（单通道），H=action_horizon，W=action_dim
- 这样可以使用标准的 CNN encoder/decoder

### 为什么 transpose 失效？

```python
# 输入: (1, 1, 16, 7)
# transpose(0, 1): 交换维度 0 和 1
# 输出: (1, 1, 16, 7)  ← 因为维度 0 和 1 都是 1，所以交换后形状不变!
```

### 为什么队列只有 1 个元素？

```python
# extend 迭代第一个维度（长度为 1）
self._action_queue.extend(tensor_of_shape_[1,1,16,7])
# 结果：队列中添加 1 个 (1, 16, 7) 的 tensor
```

## 解决方案

### 修复位置

**文件**: `src/lerobot/policies/carp/modeling_carp.py`

**方法**: `predict_action_chunk()` (line 256-295)

### 修改内容

```python
@torch.no_grad()
def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
    """
    Returns:
        actions: (B, action_horizon, action_dim)
    """
    # ... 原有代码 ...

    # Autoregressive generation using the AR model's inference method
    actions = self.ar_model.autoregressive_infer_cfg(
        nobs=obs_dict,
        vae_proxy=self.vae,
        ntasks=task_ids,
    )  # Returns (B, 1, action_horizon, action_dim) from VAE

    # ========== 新增代码 ==========
    # Remove the extra dimension (dimension 1)
    # VAE's fhat_to_action returns [B, 1, action_horizon, action_dim]
    # We need [B, action_horizon, action_dim]
    actions = actions.squeeze(1)  # (B, action_horizon, action_dim)

    return actions
```

**关键**: 添加 `squeeze(1)` 移除维度 1（值为 1 的维度）

## 修复后的数据流

```
predict_action_chunk:
  ar_model.autoregressive_infer_cfg
    ↓
  vae.fhat_to_action: [B, 1, action_horizon, action_dim]
    ↓ squeeze(1)
  [B, action_horizon, action_dim]  ← 正确的 3 维
    ↓ 返回

select_action:
  predict_action_chunk: [1, 16, 7]
    ↓ [:, :action_horizon, :]
  [1, 16, 7]
    ↓ transpose(0, 1)
  [16, 1, 7]  ← 现在 transpose 有效了！
    ↓ extend
  queue: [(1, 7), (1, 7), ..., (1, 7)]  ← 16 个元素
    ↓ popleft
  [1, 7]  ✓ 正确！
```

## 与训练的一致性

### 训练时的处理

**文件**: `src/lerobot/policies/carp/modeling_carp.py` line 165-171

```python
def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    # Extract actions
    actions = batch["action"]  # (B, T, A)

    # VAE expects (B, 1, T, A) format
    actions_vae_format = actions.unsqueeze(1)  # (B, 1, T, A)

    # Use VAE to encode
    gt_token_indices = self.vae.inp_to_idxBl(actions_vae_format, ...)
```

**训练时**：
- 输入：`(B, T, A)`
- VAE 输入：`unsqueeze(1)` → `(B, 1, T, A)`
- VAE 编码/解码都使用 4 维

**推理时**（修复后）：
- AR model + VAE 输出：`(B, 1, T, A)`
- `squeeze(1)` → `(B, T, A)`
- 符合 LeRobot 框架期望

## 验证

### 预期的调试输出（修复后）

```
[select_action] predict_action_chunk returned: torch.Size([1, 16, 7])  ✓
[select_action] after slice: torch.Size([1, 16, 7])  ✓
[select_action] after transpose: torch.Size([16, 1, 7])  ✓
[select_action] queue length after extend: 16  ✓
[select_action] first queue element shape: torch.Size([1, 7])  ✓
[select_action] returning action: torch.Size([1, 7])  ✓
*****Action shape: torch.Size([1, 7]) *****  ✓
```

### 测试命令

```bash
lerobot-eval \
  --policy.path=/path/to/carp/checkpoint \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=50
```

**预期结果**：
- ✅ Action shape 是 `(1, 7)` 而不是 `(1, 16, 7)`
- ✅ 与向量化环境兼容
- ✅ 评估正常运行

## 影响范围

| 组件 | 是否受影响 | 说明 |
|------|-----------|------|
| 训练 (forward) | ✗ | 不受影响，已经正确处理 4 维 |
| 推理 (predict_action_chunk) | ✓ | 主要修复点 |
| select_action | ✓ | 依赖修复后正常工作 |
| VAE 模型 | ✗ | 不需要修改 |
| AR 模型 | ✗ | 不需要修改 |

## 其他相关修复

这个修复与之前的修复配合：
1. ✅ Task ID 映射 (`libero_task_mapping.py`)
2. ✅ ProcessorStep 注册 (`policies/carp/__init__.py`)
3. ✅ RenameObservationsProcessor 包含 (`processor_carp.py`)
4. ✅ Task ID 类型转换 (`envs/utils.py`)
5. ✅ **Action shape 修复** (`modeling_carp.py` - 本次修复)

## 总结

| 方面 | 修复前 | 修复后 |
|------|--------|--------|
| VAE 输出形状 | `(B, 1, H, A)` | `(B, 1, H, A)` (不变) |
| predict_action_chunk 输出 | `(B, 1, H, A)` | `(B, H, A)` ✓ |
| select_action 输出 | `(B, H, A)` | `(B, A)` ✓ |
| 与框架兼容性 | ✗ | ✓ |

**关键点**：
- VAE 使用 `(B, 1, H, A)` 格式是正确的（CNN "图像"格式）
- `predict_action_chunk` 需要 squeeze 掉额外维度以符合 LeRobot 约定
- 训练和推理现在都正确处理 VAE 的 4 维格式

---

**修复时间**: 2026-03-23
**修复文件**: `src/lerobot/policies/carp/modeling_carp.py` line 289
**严重程度**: 关键（阻塞评估）
**修复难度**: 简单（单行 squeeze）
