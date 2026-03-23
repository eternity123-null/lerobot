# CARP LeRobot 集成完成报告

## 项目目标

将 CARP（Coarse-to-Fine Autoregressive Policy）模型完全集成到 LeRobot 框架，支持：
- ✓ 使用 `lerobot-train` 进行标准训练
- ✓ 使用 `lerobot-eval` 进行 LIBERO 环境评估
- ✓ 保持与原始 CARP 实现的架构和超参数一致性
- ✓ 正确处理多任务学习的 task_id 注入

## 完成的工作

### 1. 核心实现 ✓

| 文件 | 功能 | 状态 |
|------|------|------|
| `policies/carp/configuration_carp.py` | CARP 配置类 | ✓ 完成 |
| `policies/carp/modeling_carp.py` | CARP Policy 实现 | ✓ 完成 |
| `policies/carp/processor_carp.py` | 数据预处理/后处理 | ✓ 完成 |
| `policies/carp/vae_model.py` | VAE (MSAT) 模型 | ✓ 完成 |
| `policies/carp/ar_model.py` | AR (CFAP) 模型 | ✓ 完成 |
| `policies/carp/carp_processor_steps.py` | 自定义处理步骤 | ✓ 完成 |
| `policies/carp/carp_task_id_step.py` | Task ID 处理步骤 | ✓ 完成 |
| `policies/factory.py` | 工厂注册 | ✓ 完成 |
| `policies/__init__.py` | 配置类导出 | ✓ 完成 |

### 2. 框架增强 ✓

| 文件 | 修改内容 | 目的 |
|------|---------|------|
| `processor/converters.py` | 添加 task_id 提取 | 支持 task_id 在 complementary_data 中传递 |
| `envs/utils.py` | 添加 task_id 到观测 | 评估时自动从环境获取 task_id |

### 3. 架构验证 ✓

**参考文档**: `CARP_ARCHITECTURE_VERIFICATION.md`

| 组件 | 原始 CARP | LeRobot CARP | 差异 |
|------|-----------|--------------|------|
| VAE 参数量 | 101.48K | 101.48K | 0% |
| AR 参数量 | 26.31M (8 tasks) | 26.30M (40 tasks) | 0.02% |
| Core Transformer | 14.82M | 14.82M | 0% |
| Vocab Size | 1024 | 1024 | ✓ 匹配 |
| AR Depth | 32 | 32 | ✓ 匹配 |
| AR Embed Dim | 160 | 160 | ✓ 匹配 |
| AR Num Heads | 32 | 32 | ✓ 匹配 |
| AR MLP Ratio | 4.0 | 4.0 | ✓ 匹配 |

**结论**: 所有核心架构参数完全匹配 ✓

### 4. 超参数验证 ✓

**参考文档**: `CARP_TRAINING_COMMANDS.md`

| 参数 | 原始 CARP | LeRobot CARP | 状态 |
|------|-----------|--------------|------|
| VAE Batch Size | 256 | 256 | ✓ |
| VAE Learning Rate | 3e-4 | 3e-4 | ✓ |
| VAE Steps | 200k | 200k | ✓ |
| AR Batch Size | 64 | 64 | ✓ |
| AR Learning Rate | 3e-4 | 3e-4 | ✓ |
| AR Steps | 200k | 200k | ✓ |
| Action Horizon | 16 | 16 | ✓ |
| n_obs_steps | 1 | 1 | ✓ |
| Patch Nums | (1,2,3,4) | (1,2,3,4) | ✓ |

**结论**: 所有训练超参数完全匹配 ✓

### 5. Task ID 注入验证 ✓

**参考文档**: `TASK_ID_INJECTION_COMPLETE.md`

#### 训练流程
```
LeRobotDataset[task_index]
  → CARPTaskIDProcessorStep
  → batch[task_id]
  → CARPPolicy.forward(task_ids)
```

**测试**: `test_train_task_id.py` ✓ 通过

#### 评估流程
```
LiberoEnv.task_id
  → add_envs_task()
  → obs[task_id]
  → CARPTaskIDProcessorStep
  → batch[task_id]
  → CARPPolicy.select_action(task_ids)
```

**测试**: `test_task_id_flow.py` ✓ 通过

### 6. 文档输出 ✓

| 文档 | 内容 |
|------|------|
| `CARP_ARCHITECTURE_VERIFICATION.md` | 架构参数对比分析 |
| `CARP_TRAINING_COMMANDS.md` | 训练命令和超参数说明 |
| `CARP_TASK_ID_INJECTION.md` | Task ID 注入机制详解 |
| `TASK_ID_INJECTION_COMPLETE.md` | Task ID 完整验证报告 |
| `INTEGRATION_COMPLETE.md` | 本文档（总结报告） |

## 使用指南

### 阶段 1: 训练 VAE (MSAT)

```bash
lerobot-train \
  --policy=carp \
  --dataset.repo_id=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
  --policy.ar_training_mode=false \
  --policy.vae_checkpoint_path=null \
  --policy.device=cuda \
  --batch_size=256 \
  --steps=200000 \
  --eval_freq=10000 \
  --save_freq=10000 \
  --output_dir=outputs/carp_vae_libero \
  --policy.n_obs_steps=1 \
  --policy.action_horizon=16 \
  --optimizer.lr=0.0003 \
  --optimizer.weight_decay=0.0001 \
  --wandb.enable=true \
  --wandb.project=carp-lerobot
```

### 阶段 2: 训练 AR (CFAP)

```bash
lerobot-train \
  --policy=carp \
  --dataset.repo_id=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
  --policy.ar_training_mode=true \
  --policy.vae_checkpoint_path=outputs/carp_vae_libero/checkpoints/200000/pretrained_model \
  --policy.device=cuda \
  --batch_size=64 \
  --steps=200000 \
  --eval_freq=10000 \
  --save_freq=10000 \
  --output_dir=outputs/carp_ar_libero \
  --policy.n_obs_steps=1 \
  --policy.action_horizon=16 \
  --optimizer.lr=0.0003 \
  --optimizer.weight_decay=0.0001 \
  --wandb.enable=true \
  --wandb.project=carp-lerobot
```

### 评估

```bash
lerobot-eval \
  --policy.path=outputs/carp_ar_libero/checkpoints/200000/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=50 \
  --eval.batch_size=10 \
  --policy.device=cuda
```

## 技术亮点

### 1. 两阶段训练架构

- **Stage 1 (VAE)**: 学习动作序列的紧凑表示（multi-scale VQ tokens）
- **Stage 2 (AR)**: 学习从观测到 tokens 的自回归预测

### 2. Multi-Scale Action Tokenization (MSAT)

- Per-dimension VQ-VAE，每个动作维度独立量化
- 4 个尺度（patch_nums=1,2,3,4），提供从粗到细的预测
- 总 codebook 大小：7 (dims) × 4 (scales) × 1024 (vocab) = 28,672 tokens

### 3. Coarse-to-Fine Autoregressive Prediction (CFAP)

- 32-layer Transformer，160 embedding dim，32 attention heads
- 逐 token 自回归生成：先粗粒度（scale 0），后细粒度（scale 3）
- 支持多任务条件化：task embedding 与观测 concat

### 4. Task ID 注入机制

- **训练**: 数据集 task_index → CARPTaskIDProcessorStep → model task_id
- **评估**: 环境 task_id → add_envs_task → CARPTaskIDProcessorStep → model task_id
- **兼容性**: 不影响其他策略，向后兼容

## 测试验证

| 测试 | 文件 | 结果 |
|------|------|------|
| 训练 task_id 注入 | `test_train_task_id.py` | ✓ 通过 |
| 评估 task_id 注入 | `test_task_id_flow.py` | ✓ 通过 |
| 架构参数匹配 | `check_ar_params_detailed.py` | ✓ 通过 |

## 代码质量

- ✓ 遵循 LeRobot 代码规范
- ✓ 完整的类型注解
- ✓ 详细的文档字符串
- ✓ 注册表模式集成
- ✓ Hub 兼容（save_pretrained/from_pretrained）

## 未来工作

1. **性能优化**:
   - 实现 flash attention（需要 flash-attn 库）
   - 实现 torch.compile 支持
   - 优化 VQ 查找的 GPU 效率

2. **功能扩展**:
   - 支持语言条件（可选）
   - 支持更多环境（MetaWorld, RLBench）
   - 实现 curriculum learning

3. **实验验证**:
   - LIBERO 90 任务的完整训练和评估
   - 与其他 LeRobot 策略的性能对比
   - 消融实验（scales, depth, embedding dim）

## 总结

CARP 模型已经**完全集成**到 LeRobot 框架中：

- ✅ 架构参数与原始 CARP 完全一致
- ✅ 训练超参数与原始 CARP 完全匹配
- ✅ Task ID 注入机制完整实现并验证
- ✅ 支持标准的 lerobot-train/lerobot-eval 命令
- ✅ 所有测试通过

用户可以立即开始使用 CARP 进行 LIBERO 多任务学习实验。

---

**集成完成时间**: 2026-03-22
**LeRobot 版本**: latest
**CARP 论文**: [CARP: Coarse-to-Fine Autoregressive Policy]
