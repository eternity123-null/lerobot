# CARP (Coarse-to-Fine Autoregressive Policy) 集成到 LeRobot

本文档介绍了如何使用集成到 LeRobot 框架中的 CARP 模型。

## 概述

CARP 是一个两阶段训练的机器人策略模型：

1. **Stage 1: Multi-Scale Action Tokenization (MSAT)**
   - 训练 VQ-VAE 将动作序列编码为离散的多尺度 token
   - 输入：动作序列 (B, action_horizon, action_dim)
   - 输出：重建动作 + VQ 损失

2. **Stage 2: Coarse-to-Fine Autoregressive Prediction (CFAP)**
   - 训练 Transformer 从观测预测动作 token
   - 输入：观测（图像 + 状态）+ 任务 ID
   - 输出：预测的动作 token（通过冻结的 VAE 解码为动作）

## 文件结构

```
src/lerobot/policies/carp/
├── configuration_carp.py      # CARP 配置类
├── modeling_carp_vae.py       # Stage 1: VAE 训练策略
├── modeling_carp.py           # Stage 2: AR 训练策略
├── processor_carp.py          # 数据处理器
├── obs_encoder_builder.py    # 观测编码器（ResNet18）
├── MSAT/                      # Multi-Scale Action Tokenizer
│   ├── vqvae.py              # VQ-VAE 实现
│   └── quant.py              # Vector Quantizer
├── CFAP/                      # Coarse-to-Fine AR
│   ├── autoreg.py            # Transformer 实现
│   └── basic_ar.py           # 基础组件
└── ...

train_carp_vae.py             # Stage 1 训练脚本
train_carp_ar.py              # Stage 2 训练脚本
```

## 测试脚本

```
test_carp_integration.py         # 基础集成测试
test_carp_libero_integration.py  # LIBERO 数据集集成测试
test_vae_training.py             # VAE 训练循环测试
```

## 使用方法

### 1. 数据集准备

LIBERO 数据集路径：
```
/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero
```

数据集包含：
- 273,465 帧数据
- 多个 episode
- 观测：2 个相机（image, image2）+ 8 维状态
- 动作：7 维

### 2. Stage 1: 训练 VQ-VAE

```bash
python train_carp_vae.py \\
    --dataset_path=/path/to/libero \\
    --output_dir=outputs/carp_vae \\
    --batch_size=32 \\
    --num_epochs=100 \\
    --learning_rate=3e-4
```

输出：
- `outputs/carp_vae/vae_final.pt` - 最终 VAE 模型
- `outputs/carp_vae/checkpoint_epoch_*.pt` - 训练 checkpoint

### 3. Stage 2: 训练 AR Transformer

```bash
python train_carp_ar.py \\
    --dataset_path=/path/to/libero \\
    --vae_checkpoint=outputs/carp_vae/vae_final.pt \\
    --output_dir=outputs/carp_ar \\
    --batch_size=32 \\
    --num_epochs=100 \\
    --learning_rate=1e-4
```

输出：
- `outputs/carp_ar/ar_final.pt` - 最终 AR 模型
- `outputs/carp_ar/checkpoint_epoch_*.pt` - 训练 checkpoint

### 4. 评估（TODO）

```bash
lerobot-eval \\
    --policy.path=outputs/carp_ar/ar_final.pt \\
    --env.type=libero \\
    --env.task=libero_object \\
    --eval.n_episodes=10
```

## 配置参数

### VAE 配置

```python
CARPConfig(
    training_stage="vae",
    action_horizon=16,      # 动作序列长度
    vocab_size=512,         # codebook 大小
    vocab_ch=8,             # 潜在空间维度
    vch=2,                  # VAE 基础通道数
    vae_lr=3e-4,            # 学习率
    vae_dropout=0.0,        # Dropout
    vqbeta=0.25,            # VQ commitment loss 权重
    vqnorm=True,            # 使用余弦相似度
    vqresi=0.5,             # 残差连接比例
)
```

### AR 配置

```python
CARPConfig(
    training_stage="ar",
    action_horizon=16,
    n_obs_steps=1,          # 观测历史长度
    ar_depth=32,            # Transformer 层数
    ar_embed_dim=160,       # 嵌入维度
    ar_num_heads=32,        # 注意力头数
    ar_lr=1e-4,             # 学习率
    ar_dropout=0.0,         # Dropout
    task_num=10,            # 任务数量
    vae_checkpoint_path="path/to/vae.pt",  # 预训练 VAE 路径
)
```

## 测试

运行所有测试：

```bash
# 基础集成测试
python test_carp_integration.py

# LIBERO 数据集测试
python test_carp_libero_integration.py

# VAE 训练测试
python test_vae_training.py
```

所有测试应该通过：
```
✓ 配置注册
✓ 策略导入
✓ 处理器创建
✓ VAE模型创建
✓ 数据集加载
✓ 特征兼容性
✓ DataLoader集成
✓ VAE 训练循环
```

## 已知限制

1. **动作序列采样**：当前实现简化了动作序列的处理，实际训练时需要正确采样 future actions
2. **观测历史**：AR 训练脚本需要实现多步观测历史（n_obs_steps）
3. **分布式训练**：部分代码需要初始化分布式环境，当前已修复为支持单 GPU 训练
4. **LIBERO 评估**：评估集成尚未完成

## 技术细节

### 观测编码器

- 使用 ResNet18 作为视觉编码器（每个相机独立）
- 冻结 BatchNorm 层
- 输出：每个相机 128 维特征
- 状态：直接拼接（Identity mapping）

### VQ-VAE 架构

- Per-dimension VQ-VAE：每个动作维度独立编码
- 多尺度：patch_nums=(1, 2, 3, 4)
- 残差连接：支持部分共享 residual function
- 量化方法：余弦相似度（可选欧氏距离）

### AR Transformer

- Adaptive Layer Normalization (AdaLN)
- Coarse-to-fine 生成：从粗粒度到细粒度
- Teacher forcing 训练
- Autoregressive 推理

## 后续工作

1. [ ] 实现完整的动作序列采样
2. [ ] 实现观测历史处理
3. [ ] 完成 LIBERO 评估集成
4. [ ] 优化训练效率
5. [ ] 添加 Weights & Biases 日志记录
6. [ ] 支持分布式训练
7. [ ] 添加学习率调度器
8. [ ] 实现 checkpointing 和恢复训练

## 参考

- CARP 原始论文: [链接待添加]
- LeRobot 文档: https://github.com/huggingface/lerobot
- LIBERO 基准测试: https://github.com/Lifelong-Robot-Learning/LIBERO
