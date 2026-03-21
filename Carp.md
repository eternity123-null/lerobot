# CARP Multitask 模型在 LeRobot 框架中的复现计划

> **文档版本**: v1.0
> **创建日期**: 2026-03-21
> **预计完成时间**: 今晚

---

## 📋 目录

- [核心目标](#核心目标)
- [技术架构分析](#技术架构分析)
- [复现策略](#复现策略)
- [详细实现计划](#详细实现计划)
- [关键风险和应对策略](#关键风险和应对策略)
- [时间规划和里程碑](#时间规划和里程碑)
- [验证标准](#验证标准)
- [参考资源](#参考资源)

---

## 🎯 核心目标

将 CARP (Coarse-to-Fine Autoregressive Policy) multitask 模型集成到 LeRobot 框架中，使其能够：

1. ✅ 使用 LeRobot 的标准训练流程和数据格式 (LeRobotDataset)
2. ✅ 在 LIBERO 环境中进行评估和测试
3. ✅ 保持 CARP 原有的技术特性（两阶段训练、多尺度自回归）
4. ✅ 符合 LeRobot 的代码规范和架构模式
5. 完成关键步骤实现后及时push到当前所在lerobot repo的carp分支上

---

## 🔍 技术架构分析

### CARP 核心特点

#### 1. 两阶段训练范式

**Stage 1: MSAT (Multi-Scale Action Tokenization)**
- **Per-dimension VQ-VAE**: 为每个动作维度训练独立的 encoder/decoder/quantizer
- **多尺度量化**: v_patch_nums=(1,2,3,4) 定义 4 个粒度级别
- **输入/输出**: [B, 1, 16, act_dim] → 多尺度 token indices
- **关键组件**:
  - `MultiScaleActionTokenizer`: 10个并行 VQ-VAE (multitask)
  - `VectorQuantizer2`: cosine similarity 量化器
  - Encoder/Decoder: 卷积网络

**Stage 2: CFAP (Coarse-to-Fine Autoregressive Prediction)**
- **Transformer 架构**: depth=32, embed_dim=160, num_heads=32
- **AdaLN**: 条件化在观察上的自适应层归一化
- **从粗到精预测**: 逐尺度预测动作 token
- **关键组件**:
  - `Coarse2FineAutoRegressor`: 自回归 Transformer
  - obs_encoder: robomimic 的 ObservationEncoder
  - task_embed: 任务嵌入层 (nn.Embedding)

#### 2. 数据流

```
HDF5数据 (robomimic格式)
    ↓
BaseImageDataset
    ↓
(obs, action) → Normalize
    ↓
Stage 1: action → VAE → indices (多尺度token)
    ↓
Stage 2: (obs, indices) → AR Transformer → pred_logits
    ↓
Inference: obs → AR → pred_indices → VAE.decode() → action
    ↓
Environment Rollout
```

#### 3. 关键技术参数

| 参数类别 | 参数名 | 默认值 | 说明 |
|---------|--------|--------|------|
| **MSAT** | vocab_size | 512 | codebook 大小 |
| | vocab_ch (z_channels) | 8 | latent 维度 |
| | vch | 2 | 基础通道数 |
| | ch_mult | (2, 4) | 下采样倍数 |
| | vqbeta | 0.25 | commitment loss weight |
| | vqnorm | True | cosine similarity |
| | vae_lr | 3e-4 | VAE 学习率 |
| **CFAP** | ar_depth | 32 | Transformer 层数 |
| | ar_embed_dim | 160 | embedding 维度 |
| | ar_num_heads | 32 | 注意力头数 |
| | ar_lr | 1e-4 | AR 学习率 |
| | ar_weight_decay | 0.05 | 权重衰减 |
| **多尺度** | patch_nums | (1,2,3,4) | 4个尺度 |
| | action_horizon | 16 | 动作序列长度 |
| **多任务** | task_num | 8 | LIBERO 任务数量 |
| | task_embed_dim | 3 | 任务嵌入维度 |

### LeRobot 框架要求

#### 1. 标准三组件模式

每个策略必须包含：

```
src/lerobot/policies/{policy_name}/
├── configuration_{policy}.py  # PreTrainedConfig 子类
├── modeling_{policy}.py       # PreTrainedPolicy 子类
└── processor_{policy}.py      # make_*_pre_post_processors() 工厂函数
```

#### 2. 核心接口

```python
class PreTrainedPolicy(nn.Module):
    # 训练时调用
    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        """返回 (loss, loss_dict)"""
        pass

    # 推理时调用
    @torch.no_grad()
    def select_action(self, batch: dict) -> Tensor:
        """返回单步动作"""
        pass

    # 环境重置时调用
    def reset(self):
        """清理内部状态"""
        pass

    # 返回可优化参数
    def get_optim_params(self) -> dict:
        """返回 parameters()"""
        pass
```

#### 3. 数据格式规范

- **LeRobotDataset**: Parquet + MP4/images
- **特征命名约定**:
  - `observation.images.{cam_name}`: 图像观察
  - `observation.state`: 状态观察
  - `action`: 动作
- **ProcessorPipeline**: 前处理 + 后处理流水线

---

## 🏗️ 复现策略：混合方法

### 核心思想

**保留 CARP 核心实现 + 创建 LeRobot 适配层**

### 架构设计

```
┌─────────────────────────────────────────────────────┐
│         LeRobot Training Script                     │
│         (lerobot-train / lerobot-eval)              │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         CARPPolicy (LeRobot Interface Layer)        │
│  - configuration_carp.py                            │
│  - modeling_carp.py (CARPPolicy, CARPVAEPolicy)     │
│  - processor_carp.py                                │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         CARP Core Implementation                    │
│  - MSAT/ (MultiScaleActionTokenizer)                │
│  - CFAP/ (Coarse2FineAutoRegressor)                 │
│  - svqvae/ (Per-dimension VQ-VAE)                   │
│  - optim/ (AmpOptimizer, lr_control)                │
│  - utils/ (arg_util, train_util, etc.)              │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         LeRobotDataset + Environment                │
│  - LeRobotDataset (Parquet + MP4)                   │
│  - LiberoEnv (LIBERO benchmark)                     │
└─────────────────────────────────────────────────────┘
```

### 优势

- ✅ **快速复现**: 最小化对 CARP 核心代码的修改
- ✅ **降低风险**: 保留原始实现的正确性
- ✅ **易于维护**: 清晰的分层设计
- ✅ **支持特殊流程**: 两阶段训练流程
- ✅ **便于对比**: 可与原始 CARP 实现对比验证

---

## 📝 详细实现计划

### 阶段 1: 环境准备和依赖管理 (0.5天)

#### 任务 1.1: 分析依赖冲突

**目标**: 确保 CARP 和 LeRobot 的依赖兼容

**步骤**:
1. 比较 `carp/multitask/environment.yaml` vs LeRobot `pyproject.toml`
2. 识别版本冲突：torch, numpy, gymnasium, robomimic
3. 决策：在 carp_le 环境中安装最小 CARP 依赖

**关键依赖检查**:
```bash
# 检查当前环境
conda activate carp_le
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"

# 检查 lerobot 是否可用
python -c "import lerobot; print('LeRobot OK')"
```

#### 任务 1.2: 准备开发环境

```bash
# 在 carp_le 环境中
conda activate carp_le

# 仅安装 CARP 必需的额外依赖
pip install typed-argument-parser  # for arg_util.py

# 不要安装完整的 environment.yaml，避免破坏 lerobot 环境
```

**交付物**:
- [ ] 依赖兼容性报告 (dependency_report.md)
- [ ] 环境配置文档 (environment_setup.md)

---

### 阶段 2: CARP 核心代码集成 (1天)

#### 任务 2.1: 复制 CARP 核心模块

```bash
# 创建 carp policy 目录
mkdir -p src/lerobot/policies/carp/

# 复制核心模块 (保持原始实现)
cp -r carp/multitask/MSAT src/lerobot/policies/carp/
cp -r carp/multitask/CFAP src/lerobot/policies/carp/
cp -r carp/multitask/svqvae src/lerobot/policies/carp/
cp carp/multitask/dist.py src/lerobot/policies/carp/

# 复制辅助模块
cp -r carp/multitask/optim src/lerobot/policies/carp/
mkdir -p src/lerobot/policies/carp/utils
cp carp/multitask/utils/arg_util.py src/lerobot/policies/carp/utils/
cp carp/multitask/utils/train_util.py src/lerobot/policies/carp/utils/
cp carp/multitask/utils/helpers.py src/lerobot/policies/carp/utils/

# 创建 __init__.py
touch src/lerobot/policies/carp/__init__.py
touch src/lerobot/policies/carp/MSAT/__init__.py
touch src/lerobot/policies/carp/CFAP/__init__.py
touch src/lerobot/policies/carp/svqvae/__init__.py
touch src/lerobot/policies/carp/optim/__init__.py
touch src/lerobot/policies/carp/utils/__init__.py
```

#### 任务 2.2: 解决导入路径问题

**需要修改的导入**:

```python
# 修改前 (CARP 原始)
from MSAT import MultiScaleActionTokenizer
from CFAP import Coarse2FineAutoRegressor
from svqvae.basic_vae import Encoder, Decoder

# 修改后 (LeRobot)
from lerobot.policies.carp.MSAT.vqvae import MultiScaleActionTokenizer
from lerobot.policies.carp.CFAP.autoreg import Coarse2FineAutoRegressor
from lerobot.policies.carp.svqvae.basic_vae import Encoder, Decoder
```

**涉及的文件**:
- `MSAT/vqvae.py`: 修改 `from svqvae.basic_vae import ...`
- `CFAP/autoreg.py`: 修改 `from MSAT.vqvae import ...`
- 所有其他相对导入

#### 任务 2.3: 移除不必要的依赖

**删除或保留决策**:

| 模块 | 决策 | 原因 |
|------|------|------|
| `env/dataset/` | ❌ 删除 | LeRobot 有自己的数据集系统 |
| `env/runner/` | ❌ 删除 | LeRobot 有自己的评估系统 |
| `env/robomimic/` | ❌ 删除 | 不需要 robomimic 环境包装 |
| `env/common/normalizer.py` | ✅ 保留 | 可能需要复用归一化逻辑 |
| `env/common/rotation_transformer.py` | ✅ 保留 | 旋转表示转换工具 |
| `env/common/pytorch_util.py` | ✅ 保留 | dict_apply 等通用工具 |
| `trainer_ar.py` | ❌ 删除 | 使用 LeRobot 的训练循环 |
| `trainer_vae.py` | ❌ 删除 | 使用 LeRobot 的训练循环 |

**交付物**:
- [ ] `src/lerobot/policies/carp/MSAT/` (完整)
- [ ] `src/lerobot/policies/carp/CFAP/` (完整)
- [ ] `src/lerobot/policies/carp/optim/` (完整)
- [ ] `src/lerobot/policies/carp/utils/` (部分)
- [ ] 导入路径修复清单 (import_fixes.md)

---

### 阶段 3: 创建 LeRobot Policy 配置类 (0.5天)

#### 任务 3.1: 创建 configuration_carp.py

**文件**: `src/lerobot/policies/carp/configuration_carp.py`

**核心结构**:

```python
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig

@PreTrainedConfig.register_subclass("carp")
@dataclass
class CARPConfig(PreTrainedConfig):
    """
    CARP (Coarse-to-Fine Autoregressive Policy) 配置

    分为两个训练阶段:
    1. MSAT (Multi-Scale Action Tokenization): 训练 VQ-VAE
    2. CFAP (Coarse-to-Fine Autoregressive Prediction): 训练 Transformer
    """

    # ============ 通用配置 ============
    n_obs_steps: int = 1
    action_horizon: int = 16

    # ============ MSAT (VQ-VAE) 配置 ============
    vocab_size: int = 512
    vocab_ch: int = 8
    vch: int = 2
    ch_mult: tuple[int, ...] = (2, 4)
    vae_dropout: float = 0.0
    vqbeta: float = 0.25
    vqnorm: bool = True
    vqresi: float = 0.5
    vae_lr: float = 3e-4
    vae_weight_decay: float = 0.005

    # ============ CFAP (Transformer) 配置 ============
    ar_depth: int = 32
    ar_embed_dim: int = 160
    ar_num_heads: int = 32
    ar_mlp_ratio: float = 4.0
    ar_dropout: float = 0.0
    ar_lr: float = 1e-4
    ar_weight_decay: float = 0.05

    # ============ 多尺度配置 ============
    patch_nums: tuple[int, ...] = (1, 2, 3, 4)

    # ============ 多任务配置 ============
    task_num: int = 8
    task_embed_dim: int = 3

    # ============ 训练阶段控制 ============
    training_stage: str = "ar"  # "vae" 或 "ar"
    vae_checkpoint_path: str | None = None

    # ============ 归一化配置 ============
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def get_optimizer_preset(self) -> AdamWConfig:
        if self.training_stage == "vae":
            return AdamWConfig(lr=self.vae_lr, weight_decay=self.vae_weight_decay)
        else:
            return AdamWConfig(lr=self.ar_lr, weight_decay=self.ar_weight_decay)

    def validate_features(self) -> None:
        assert self.input_features is not None
        assert self.output_features is not None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[list[int]]:
        return [[0, self.action_horizon]]
```

**关键设计要点**:
1. 使用 `@PreTrainedConfig.register_subclass("carp")` 注册
2. 包含 VAE 和 AR 两个阶段的所有参数
3. `training_stage` 控制当前训练哪个阶段
4. 实现所有抽象方法

**交付物**:
- [ ] `src/lerobot/policies/carp/configuration_carp.py`
- [ ] 配置参数文档 (config_params.md)

---

### 阶段 4: 创建 LeRobot Policy 模型类 (2天)

#### 任务 4.1: 创建 modeling_carp_vae.py

**文件**: `src/lerobot/policies/carp/modeling_carp_vae.py`

**用途**: Stage 1 训练 Multi-Scale Action Tokenizer

**核心代码框架**:

```python
import torch
import torch.nn as nn
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.carp.configuration_carp import CARPConfig
from lerobot.policies.carp.MSAT.vqvae import MultiScaleActionTokenizer

class CARPVAEPolicy(PreTrainedPolicy):
    """CARP Stage 1: 训练 VQ-VAE"""
    config_class = CARPConfig
    name = "carp_vae"

    def __init__(self, config: CARPConfig):
        super().__init__(config)
        assert config.training_stage == "vae"

        action_dim = config.output_features["action"].shape[0]

        self.vae = MultiScaleActionTokenizer(
            vocab_size=config.vocab_size,
            z_channels=config.vocab_ch,
            ch=config.vch,
            ch_mult=config.ch_mult,
            action_dim=action_dim,
            num_actions=config.action_horizon,
            dropout=config.vae_dropout,
            beta=config.vqbeta,
            using_znorm=config.vqnorm,
            quant_resi=config.vqresi,
            v_patch_nums=config.patch_nums,
            test_mode=False,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """训练 VQ-VAE"""
        actions = batch["action"]  # (B, T, A)
        actions = actions.unsqueeze(1)  # (B, 1, T, A)

        recon_actions, usages, vq_loss = self.vae(actions, ret_usages=True)
        recon_loss = torch.nn.functional.mse_loss(recon_actions, actions)
        total_loss = recon_loss + vq_loss

        loss_dict = {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "commit_loss": vq_loss.item(),
        }

        return total_loss, loss_dict

    def get_optim_params(self):
        return self.vae.parameters()

    def reset(self):
        pass

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("VAE policy is not for inference")
```

#### 任务 4.2: 创建 modeling_carp.py (主策略)

**文件**: `src/lerobot/policies/carp/modeling_carp.py`

**用途**: Stage 2 训练 Coarse-to-Fine Autoregressive Model

**核心代码框架**:

```python
import torch
import torch.nn as nn
from collections import deque
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.carp.configuration_carp import CARPConfig
from lerobot.policies.carp.MSAT.vqvae import MultiScaleActionTokenizer
from lerobot.policies.carp.CFAP.autoreg import Coarse2FineAutoRegressor

class CARPPolicy(PreTrainedPolicy):
    """CARP Stage 2: Autoregressive Prediction"""
    config_class = CARPConfig
    name = "carp"

    def __init__(self, config: CARPConfig):
        super().__init__(config)
        assert config.training_stage == "ar"

        action_dim = config.output_features["action"].shape[0]

        # 1. 加载预训练的 VAE (frozen)
        self.vae = MultiScaleActionTokenizer(...)
        if config.vae_checkpoint_path:
            vae_state = torch.load(config.vae_checkpoint_path)
            self.vae.load_state_dict(vae_state["model"])

        # 冻结 VAE
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

        # 2. 创建观察编码器
        self.obs_encoder = self._build_obs_encoder()

        # 3. 创建自回归 Transformer
        self.ar_model = Coarse2FineAutoRegressor(
            vae_proxy=self.vae,
            obs_encoder=self.obs_encoder,
            action_dim=action_dim,
            task_num=config.task_num,
            depth=config.ar_depth,
            embed_dim=config.ar_embed_dim,
            patch_nums=config.patch_nums,
            n_obs_steps=config.n_obs_steps,
        )

        self._action_queue = deque(maxlen=config.action_horizon)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """训练 AR 模型"""
        actions = batch["action"]
        task_ids = batch.get("task_id", torch.zeros(actions.size(0)))

        # 1. VAE 编码 ground truth
        with torch.no_grad():
            gt_idxBls = self.vae.inp_to_idxBl(actions.unsqueeze(1))
            gt_inputs = self.vae.idxBl_to_autoreg_input(gt_idxBls)
            # ... 处理格式

        # 2. 准备观察
        nobs = self._prepare_observations(batch)

        # 3. AR 前向传播
        logits_BLV = self.ar_model(nobs, gt_inputs, task_ids)

        # 4. 计算交叉熵损失
        loss = torch.nn.functional.cross_entropy(...)

        return loss, loss_dict

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """推理"""
        if len(self._action_queue) == 0:
            nobs = self._prepare_observations(batch)
            task_ids = batch.get("task_id", torch.zeros(1))

            pred_actions = self.ar_model.autoregressive_infer_cfg(
                nobs=nobs,
                vae_proxy=self.vae,
                ntasks=task_ids,
            )

            for i in range(pred_actions.size(0)):
                self._action_queue.append(pred_actions[i])

        return self._action_queue.popleft().unsqueeze(0)
```

#### 任务 4.3: 实现观察编码器构建

**文件**: `src/lerobot/policies/carp/obs_encoder_builder.py`

**策略**: 使用 LeRobot 现有的 vision encoders

```python
from lerobot.model.vision_encoders import ResNetEncoder
import torch.nn as nn

def build_carp_obs_encoder(config: CARPConfig):
    """构建观察编码器"""
    encoders = nn.ModuleDict()
    output_dims = []

    # 图像编码器
    image_keys = [k for k in config.input_features if k.startswith("observation.images.")]
    for img_key in image_keys:
        img_shape = config.input_features[img_key].shape
        resnet = ResNetEncoder(
            input_channels=img_shape[0],
            pretrained=True,
            output_dim=128,
        )
        encoders[img_key] = resnet
        output_dims.append(128)

    # 状态编码器
    if "observation.state" in config.input_features:
        state_dim = config.input_features["observation.state"].shape[0]
        encoders["observation.state"] = nn.Identity()
        output_dims.append(state_dim)

    class CARPObsEncoder(nn.Module):
        def __init__(self, encoders, output_dims):
            super().__init__()
            self.encoders = encoders
            self._output_dim = sum(output_dims)

        def forward(self, obs_dict):
            features = []
            for key, encoder in self.encoders.items():
                if key in obs_dict:
                    feat = encoder(obs_dict[key])
                    features.append(feat)
            return torch.cat(features, dim=-1)

        def output_shape(self):
            return [self._output_dim]

    return CARPObsEncoder(encoders, output_dims)
```

**交付物**:
- [ ] `src/lerobot/policies/carp/modeling_carp_vae.py`
- [ ] `src/lerobot/policies/carp/modeling_carp.py`
- [ ] `src/lerobot/policies/carp/obs_encoder_builder.py`
- [ ] 模型单元测试

---

### 阶段 5: 创建数据处理器 (1天)

#### 任务 5.1: 创建 processor_carp.py

**文件**: `src/lerobot/policies/carp/processor_carp.py`

**核心代码**:

```python
from lerobot.processor import (
    PolicyProcessorPipeline,
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
    DeviceProcessorStep,
    AddBatchDimensionProcessorStep,
    RenameObservationsProcessorStep,
)

def make_carp_pre_post_processors(
    config: CARPConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
):
    # 预处理器
    input_steps = [
        # 1. 重命名观察键 (LeRobot → CARP)
        RenameObservationsProcessorStep(rename_map={
            "observation.images.image": "observation.images.agentview_image",
            "observation.images.image2": "observation.images.robot0_eye_in_hand_image",
        }),

        # 2. 添加 batch 维度
        AddBatchDimensionProcessorStep(),

        # 3. 归一化
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),

        # 4. 移动到设备
        DeviceProcessorStep(device=config.device),
    ]

    # 后处理器
    output_steps = [
        UnnormalizerProcessorStep(...),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline(steps=input_steps, ...),
        PolicyProcessorPipeline(steps=output_steps, ...),
    )
```

**交付物**:
- [ ] `src/lerobot/policies/carp/processor_carp.py`
- [ ] 处理器单元测试

---

### 阶段 6: 集成到 LeRobot 工厂系统 (0.5天)

#### 任务 6.1: 修改 factory.py

**文件**: `src/lerobot/policies/factory.py`

**修改点 1**: 在 `get_policy_class()` 中添加

```python
def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    # ... 现有代码 ...

    if name == "carp":
        from lerobot.policies.carp.modeling_carp import CARPPolicy
        return CARPPolicy
    elif name == "carp_vae":
        from lerobot.policies.carp.modeling_carp_vae import CARPVAEPolicy
        return CARPVAEPolicy

    # ... 现有代码 ...
```

**修改点 2**: 在 `make_pre_post_processors()` 中添加

```python
def make_pre_post_processors(policy_cfg, ...):
    # ... 现有代码 ...

    if isinstance(policy_cfg, CARPConfig):
        from lerobot.policies.carp.processor_carp import make_carp_pre_post_processors
        return make_carp_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    # ... 现有代码 ...
```

#### 任务 6.2: 修改 __init__.py

**文件**: `src/lerobot/policies/__init__.py`

```python
from lerobot.policies.carp.configuration_carp import CARPConfig

__all__ = [
    # ... 现有导出 ...
    "CARPConfig",
]
```

**交付物**:
- [ ] 修改后的 `factory.py`
- [ ] 修改后的 `__init__.py`
- [ ] 集成测试

---

### 阶段 7: 数据格式适配 (1天)

#### 任务 7.1: LeRobotDataset → CARP 格式

**关键问题**:
- LeRobotDataset 使用标准命名: `observation.images.image`, `observation.images.image2`
- CARP 期望: `agentview_image`, `robot0_eye_in_hand_image`

**解决方案**: 在 processor 中重命名 (已在阶段 5 实现)

#### 任务 7.2: 处理任务 ID

**方案 A**: 在数据集中添加 task_id 列

```python
# 修改 LeRobotDataset
dataset.add_column("task_id", task_ids)
```

**方案 B**: 在训练循环中手动添加

```python
# 在 train.py 中
for batch in dataloader:
    batch["task_id"] = torch.tensor([task_idx], device=device)
    loss, loss_dict = policy.forward(batch)
```

**推荐**: 方案 B (更简单)

#### 任务 7.3: 验证数据流

**测试脚本**: `tests/test_carp_data_flow.py`

```python
def test_data_flow():
    # 1. 加载 LeRobotDataset
    dataset = LeRobotDataset("your/dataset")

    # 2. 创建 processor
    config = CARPConfig()
    preprocessor, postprocessor = make_carp_pre_post_processors(config)

    # 3. 测试预处理
    sample = dataset[0]
    processed = preprocessor(sample)

    # 4. 验证格式
    assert "observation.images.agentview_image" in processed
    assert processed["action"].shape == (config.action_horizon, action_dim)
```

**交付物**:
- [ ] 数据格式转换文档
- [ ] 测试脚本
- [ ] 验证报告

---

### 阶段 8: 两阶段训练脚本 (1天)

#### 任务 8.1: 创建 Stage 1 训练脚本

**文件**: `scripts/train_carp_vae.sh`

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MUJOCO_GL=osmesa

# Stage 1: 训练 Multi-Scale Action Tokenizer (VAE)
python -m lerobot.scripts.train \
    --policy=carp_vae \
    --dataset.repo_id=your/libero_dataset \
    --policy.training_stage=vae \
    --policy.vocab_size=512 \
    --policy.vocab_ch=8 \
    --policy.vae_lr=3e-4 \
    --policy.vae_weight_decay=0.005 \
    --policy.action_horizon=16 \
    --batch_size=256 \
    --steps=50000 \
    --eval_freq=5000 \
    --save_freq=5000 \
    --output_dir=outputs/carp_vae \
    --wandb.enable=true \
    --wandb.project=carp_reproduction \
    --wandb.run_name=carp_vae_$(date +%Y%m%d_%H%M%S)
```

#### 任务 8.2: 创建 Stage 2 训练脚本

**文件**: `scripts/train_carp_ar.sh`

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MUJOCO_GL=osmesa

# 设置 VAE checkpoint 路径 (从 Stage 1 获取)
VAE_CKPT="outputs/carp_vae/checkpoint-50000/model.pth"

# Stage 2: 训练 Coarse-to-Fine Autoregressive Model
python -m lerobot.scripts.train \
    --policy=carp \
    --dataset.repo_id=your/libero_dataset \
    --policy.training_stage=ar \
    --policy.vae_checkpoint_path=${VAE_CKPT} \
    --policy.ar_depth=32 \
    --policy.ar_embed_dim=160 \
    --policy.ar_lr=1e-4 \
    --policy.ar_weight_decay=0.05 \
    --policy.task_num=8 \
    --batch_size=128 \
    --steps=100000 \
    --eval_freq=10000 \
    --save_freq=10000 \
    --output_dir=outputs/carp_ar \
    --wandb.enable=true \
    --wandb.project=carp_reproduction \
    --wandb.run_name=carp_ar_$(date +%Y%m%d_%H%M%S)
```

#### 任务 8.3: 创建训练配置模板

**文件**: `configs/carp_vae.yaml`

```yaml
policy:
  type: carp_vae
  training_stage: vae
  vocab_size: 512
  vocab_ch: 8
  vch: 2
  ch_mult: [2, 4]
  vae_lr: 0.0003
  vae_weight_decay: 0.005
  action_horizon: 16
  patch_nums: [1, 2, 3, 4]

training:
  batch_size: 256
  steps: 50000
  eval_freq: 5000
  save_freq: 5000

dataset:
  repo_id: your/libero_dataset

wandb:
  enable: true
  project: carp_reproduction
```

**文件**: `configs/carp_ar.yaml`

```yaml
policy:
  type: carp
  training_stage: ar
  vae_checkpoint_path: outputs/carp_vae/checkpoint-50000/model.pth
  ar_depth: 32
  ar_embed_dim: 160
  ar_num_heads: 32
  ar_lr: 0.0001
  ar_weight_decay: 0.05
  task_num: 8
  n_obs_steps: 1

training:
  batch_size: 128
  steps: 100000
  eval_freq: 10000
  save_freq: 10000
```

**交付物**:
- [ ] `scripts/train_carp_vae.sh`
- [ ] `scripts/train_carp_ar.sh`
- [ ] `configs/carp_vae.yaml`
- [ ] `configs/carp_ar.yaml`
- [ ] 训练流程文档

---

### 阶段 9: LIBERO 评估集成 (1天)

#### 任务 9.1: 验证 LIBERO 环境

```python
# test_libero_env.py
from lerobot.envs.libero import LiberoEnv

# 测试单个任务
env = LiberoEnv.from_task_name('libero_10', task_id=0)
obs = env.reset()
print(f"Observation keys: {obs.keys()}")

# 测试动作空间
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
print(f"Step successful: reward={reward}, done={done}")
```

#### 任务 9.2: 创建评估脚本

**文件**: `scripts/eval_carp_libero.sh`

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=osmesa

# 评估 CARP 在 LIBERO 上的性能
python -m lerobot.scripts.eval \
    --policy.path=outputs/carp_ar/checkpoint-100000 \
    --env.type=libero \
    --env.task=libero_10 \
    --eval.n_episodes=50 \
    --eval.batch_size=10 \
    --output_dir=eval_results/carp_libero_$(date +%Y%m%d_%H%M%S)
```

#### 任务 9.3: 多任务评估循环

**文件**: `scripts/eval_carp_multitask.py`

```python
#!/usr/bin/env python3
"""
多任务评估脚本
对 LIBERO 的所有任务进行循环评估
"""

import torch
from lerobot.policies.carp.modeling_carp import CARPPolicy
from lerobot.envs.libero import LiberoEnv

def evaluate_multitask(policy_path: str, task_suite: str = "libero_10"):
    # 加载策略
    policy = CARPPolicy.from_pretrained(policy_path)
    policy.eval()

    # LIBERO-10 有 10 个任务
    n_tasks = 10 if task_suite == "libero_10" else 8

    results = {}
    for task_id in range(n_tasks):
        print(f"\n=== Evaluating Task {task_id} ===")

        # 创建环境
        env = LiberoEnv.from_task_name(task_suite, task_id=task_id)

        # 评估
        success_count = 0
        n_episodes = 50

        for ep in range(n_episodes):
            obs = env.reset()
            done = False

            while not done:
                # 添加 task_id 到观察
                obs["task_id"] = torch.tensor([task_id])

                # 选择动作
                action = policy.select_action(obs)

                # 执行动作
                obs, reward, done, info = env.step(action)

            if info.get("success", False):
                success_count += 1

        success_rate = success_count / n_episodes
        results[f"task_{task_id}"] = success_rate
        print(f"Task {task_id} Success Rate: {success_rate:.2%}")

    # 打印汇总
    print("\n=== Summary ===")
    avg_success = sum(results.values()) / len(results)
    print(f"Average Success Rate: {avg_success:.2%}")

    for task_id, sr in results.items():
        print(f"{task_id}: {sr:.2%}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--task_suite", default="libero_10")
    args = parser.parse_args()

    evaluate_multitask(args.policy_path, args.task_suite)
```

**交付物**:
- [ ] `scripts/eval_carp_libero.sh`
- [ ] `scripts/eval_carp_multitask.py`
- [ ] 评估结果模板
- [ ] 可视化脚本

---

### 阶段 10: 测试和调试 (2天)

#### 任务 10.1: 单元测试

**文件**: `tests/policies/test_carp_config.py`

```python
import pytest
from lerobot.policies.carp.configuration_carp import CARPConfig

def test_carp_config_creation():
    config = CARPConfig()
    assert config.vocab_size == 512
    assert config.ar_depth == 32

def test_carp_config_vae_stage():
    config = CARPConfig(training_stage="vae")
    optimizer = config.get_optimizer_preset()
    assert optimizer.lr == 3e-4

def test_carp_config_ar_stage():
    config = CARPConfig(training_stage="ar")
    optimizer = config.get_optimizer_preset()
    assert optimizer.lr == 1e-4
```

**文件**: `tests/policies/test_carp_model.py`

```python
import torch
from lerobot.policies.carp.modeling_carp_vae import CARPVAEPolicy
from lerobot.policies.carp.modeling_carp import CARPPolicy

def test_vae_forward():
    config = CARPConfig(training_stage="vae")
    # 设置 features
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (10,))}

    policy = CARPVAEPolicy(config)

    batch = {
        "action": torch.randn(8, 16, 10),
    }

    loss, loss_dict = policy.forward(batch)

    assert loss.item() > 0
    assert "recon_loss" in loss_dict
    assert "commit_loss" in loss_dict

def test_ar_forward():
    config = CARPConfig(training_stage="ar", vae_checkpoint_path="dummy.pth")
    # 设置 features
    # ... (需要 mock VAE checkpoint)

    policy = CARPPolicy(config)
    # ... 测试前向传播
```

#### 任务 10.2: 集成测试

**文件**: `tests/policies/test_carp_integration.py`

```python
def test_end_to_end_vae_training():
    """端到端 VAE 训练测试"""
    # 1. 创建小型数据集
    # 2. 训练 10 steps
    # 3. 验证损失下降

def test_end_to_end_ar_training():
    """端到端 AR 训练测试"""
    # 1. 创建预训练 VAE
    # 2. 训练 AR 10 steps
    # 3. 验证损失下降

def test_inference():
    """推理测试"""
    # 1. 加载训练好的模型
    # 2. 测试 select_action
    # 3. 验证输出形状
```

#### 任务 10.3: 端到端测试

```bash
# 1. 小规模 VAE 训练
python -m lerobot.scripts.train \
    --policy=carp_vae \
    --dataset.repo_id=test_dataset \
    --steps=100 \
    --batch_size=8 \
    --output_dir=test_outputs/vae

# 2. 小规模 AR 训练
python -m lerobot.scripts.train \
    --policy=carp \
    --dataset.repo_id=test_dataset \
    --policy.vae_checkpoint_path=test_outputs/vae/checkpoint-100/model.pth \
    --steps=100 \
    --batch_size=4 \
    --output_dir=test_outputs/ar

# 3. 评估
python -m lerobot.scripts.eval \
    --policy.path=test_outputs/ar/checkpoint-100 \
    --env.type=libero \
    --env.task=libero_10 \
    --eval.n_episodes=2
```

**交付物**:
- [ ] 完整的单元测试套件
- [ ] 集成测试脚本
- [ ] 端到端测试文档
- [ ] Bug 修复清单
- [ ] 测试覆盖率报告

---

### 阶段 11: 文档和优化 (1天)

#### 任务 11.1: 编写使用文档

**文件**: `docs/policies/carp_usage.md`

```markdown
# CARP Policy 使用指南

## 简介

CARP (Coarse-to-Fine Autoregressive Policy) 是一种两阶段的视觉运动策略学习方法。

## 安装

见 README.md 的环境配置部分。

## 数据准备

### 方式 1: 使用 LeRobotDataset

如果你的数据已经是 LeRobotDataset 格式：

```bash
# 直接使用
--dataset.repo_id=your/dataset
```

**我已准备好libero的LerobotDataset数据，路径为 "/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero"**

### 方式 2: 转换现有数据

如果你有 HDF5 格式的数据：

```bash
python scripts/convert_to_lerobot_dataset.py \
    --input_path=data/raw/*.hdf5 \
    --output_path=data/lerobot_dataset
```

## 训练

### Stage 1: 训练 VAE

```bash
bash scripts/train_carp_vae.sh
```

关键参数：
- `vocab_size`: codebook 大小，默认 512
- `vocab_ch`: latent 维度，默认 8
- `vae_lr`: 学习率，默认 3e-4

### Stage 2: 训练 AR

修改 `train_carp_ar.sh` 中的 VAE checkpoint 路径：

```bash
VAE_CKPT="outputs/carp_vae/checkpoint-50000/model.pth"
```

然后运行：

```bash
bash scripts/train_carp_ar.sh
```

关键参数：
- `ar_depth`: Transformer 层数，默认 32
- `ar_embed_dim`: embedding 维度，默认 160
- `ar_lr`: 学习率，默认 1e-4

## 评估

### 单任务评估

```bash
bash scripts/eval_carp_libero.sh
```

### 多任务评估

```bash
python scripts/eval_carp_multitask.py \
    --policy_path=outputs/carp_ar/checkpoint-100000 \
    --task_suite=libero_10
```

## 常见问题

### Q: VAE 训练损失不下降？

A: 检查以下几点：
1. 学习率是否合适 (默认 3e-4)
2. beta 参数是否合适 (默认 0.25)
3. 数据归一化是否正确

### Q: AR 训练 accuracy 一直很低？

A: 确保：
1. VAE 已经充分训练（重建误差 < 0.01）
2. VAE checkpoint 路径正确
3. 观察编码器正常工作

### Q: 推理速度慢？

A: 尝试：
1. 启用 torch.compile
2. 使用 fp16/bf16
3. 增大 batch_size

## 性能基准

在 LIBERO-10 上的预期性能：

| Metric | Expected Value |
|--------|----------------|
| Average Success Rate | > 80% |
| Training Time (VAE) | ~3 hours (4x GPU) |
| Training Time (AR) | ~8 hours (4x GPU) |
| Inference Speed | < 50ms/step |
```

#### 任务 11.2: 性能优化

**优化清单**:

1. **模型编译**
```python
# 在 modeling_carp.py 中
if config.use_torch_compile:
    self.ar_model = torch.compile(self.ar_model, mode="reduce-overhead")
```

2. **混合精度训练**
```bash
# 在训练脚本中添加
--fp16=true
```

3. **数据加载优化**
```python
# 在训练配置中
num_workers=8
pin_memory=True
prefetch_factor=2
```

4. **梯度累积**
```bash
# 对于大 batch size
--batch_size=512 --gradient_accumulation_steps=4
```

#### 任务 11.3: 代码质量检查

```bash
# 格式化代码
ruff format src/lerobot/policies/carp/

# 代码检查
ruff check src/lerobot/policies/carp/

# 类型检查
mypy src/lerobot/policies/carp/
```

**交付物**:
- [ ] 完整的使用文档
- [ ] 性能优化报告
- [ ] API 参考文档
- [ ] 代码质量报告

---

### 阶段 12: 最终验证和交付 (0.5天)

#### 任务 12.1: 完整流程验证

**验证检查清单**:

- [ ] ✅ 环境配置正确
- [ ] ✅ 依赖安装成功
- [ ] ✅ 所有单元测试通过
- [ ] ✅ VAE 训练收敛
- [ ] ✅ AR 训练收敛
- [ ] ✅ LIBERO 评估成功
- [ ] ✅ 多任务评估完成
- [ ] ✅ 文档完整
- [ ] ✅ 代码符合规范

#### 任务 12.2: 性能对比

**对比 CARP 原始实现**:

| Metric | Original CARP | LeRobot CARP | Status |
|--------|---------------|--------------|--------|
| VAE Reconstruction Error | X | Y | ✅/❌ |
| AR Training Accuracy | X% | Y% | ✅/❌ |
| LIBERO-10 Success Rate | X% | Y% | ✅/❌ |
| Training Speed | X steps/s | Y steps/s | ✅/❌ |
| Inference Speed | X ms/step | Y ms/step | ✅/❌ |

#### 任务 12.3: 交付清单

**代码交付**:
- [ ] `src/lerobot/policies/carp/` (完整目录)
- [ ] 修改后的 `factory.py`
- [ ] 修改后的 `__init__.py`
- [ ] 所有测试文件
- [ ] 所有脚本文件

**文档交付**:
- [ ] `docs/policies/carp_usage.md`
- [ ] `docs/policies/carp_architecture.md`
- [ ] `docs/policies/carp_troubleshooting.md`
- [ ] API 参考文档

**配置交付**:
- [ ] `configs/carp_vae.yaml`
- [ ] `configs/carp_ar.yaml`
- [ ] `scripts/train_carp_vae.sh`
- [ ] `scripts/train_carp_ar.sh`
- [ ] `scripts/eval_carp_libero.sh`

**报告交付**:
- [ ] 实施报告
- [ ] 测试报告
- [ ] 性能对比报告
- [ ] 已知问题清单

---

## 🚨 关键风险和应对策略

| 风险 | 影响等级 | 概率 | 应对策略 | 负责阶段 |
|------|---------|------|----------|---------|
| **依赖冲突** | 🔴 高 | 中 | 最小化 CARP 依赖，仅安装必需包 | 阶段 1 |
| **观察编码器适配困难** | 🟡 中 | 高 | 先用简单 ResNet，逐步增强；准备多个备选方案 | 阶段 7 |
| **两阶段训练流程复杂** | 🟡 中 | 中 | 创建详细的训练脚本和文档；自动化流程 | 阶段 8 |
| **数据格式不兼容** | 🔴 高 | 低 | 实现转换层，充分测试验证 | 阶段 7 |
| **VAE checkpoint 加载失败** | 🔴 高 | 中 | checkpoint 兼容性检查，状态字典映射 | 阶段 4 |
| **推理性能慢** | 🟢 低 | 低 | torch.compile, 批量推理，混合精度 | 阶段 11 |
| **LIBERO 环境问题** | 🟡 中 | 中 | 提前测试环境，准备 mock 环境 | 阶段 9 |
| **内存不足** | 🟡 中 | 中 | 梯度累积，减小 batch size，混合精度 | 阶段 8 |
| **训练不稳定** | 🟡 中 | 高 | 学习率调优，梯度裁剪，warmup | 阶段 10 |

**不要动原lerobot框架已有的代码**

### 应对措施详细说明

#### 1. 依赖冲突

**预防措施**:
- 在独立环境中测试
- 使用 `pip list` 记录依赖版本
- 优先使用 conda 包管理

**应急方案**:
- 创建新的虚拟环境
- 使用 Docker 容器隔离

#### 2. 观察编码器适配

**备选方案**:
1. 方案 A: 使用 LeRobot 的 ResNetEncoder
2. 方案 B: 从 CARP 移植 robomimic ObservationEncoder
3. 方案 C: 简化版 MLP encoder (用于快速测试)

#### 3. 两阶段训练

**简化流程**:
- 创建 `train_carp_full.sh` 自动执行两阶段
- 添加 checkpoint 验证脚本
- 提供训练监控工具

---

## 📊 时间规划和里程碑

### 甘特图

```
阶段 1: 环境准备           [====]                          0.5天
阶段 2: 核心代码集成       [========]                      1天
阶段 3: 配置类             [====]                          0.5天
阶段 4: 模型类             [================]              2天
阶段 5: 处理器             [========]                      1天
阶段 6: 工厂集成           [====]                          0.5天
阶段 7: 数据适配           [========]                      1天
阶段 8: 训练脚本           [========]                      1天
阶段 9: 评估集成           [========]                      1天
阶段 10: 测试调试          [================]              2天
阶段 11: 文档优化          [========]                      1天
阶段 12: 最终验证          [====]                          0.5天
────────────────────────────────────────────────────────────
总计:                                                     12.5天
```

### 里程碑

| 里程碑 | 完成标准 | 预计完成日期 |
|--------|---------|-------------|
| **M1: 代码集成完成** | CARP 模块可导入，无导入错误 | Day 2 |
| **M2: 配置和模型就绪** | CARPConfig 和 CARPPolicy 可实例化 | Day 4.5 |
| **M3: 数据流通** | LeRobotDataset 可正常加载并处理 | Day 7.5 |
| **M4: 训练可启动** | VAE 和 AR 训练脚本可运行 | Day 8.5 |
| **M5: 评估可执行** | LIBERO 评估成功运行 | Day 9.5 |
| **M6: 测试全通过** | 所有单元测试和集成测试通过 | Day 11.5 |
| **M7: 项目交付** | 文档完整，代码符合规范 | Day 12.5 |

### 关键路径

```
环境准备 → 核心代码集成 → 模型类 → 训练脚本 → 测试调试 → 交付
```

**并行任务**:
- 阶段 5 (处理器) 和 阶段 7 (数据适配) 可部分并行
- 阶段 9 (评估) 和 阶段 11 (文档) 可部分并行

---

## ✅ 验证标准

### 阶段性验证

#### 1. 代码集成验证 (阶段 2)

```python
# 测试导入
from lerobot.policies.carp import CARPConfig
from lerobot.policies.carp.MSAT.vqvae import MultiScaleActionTokenizer
from lerobot.policies.carp.CFAP.autoreg import Coarse2FineAutoRegressor

print("✅ All imports successful")
```

**通过标准**: 无 ImportError

#### 2. 配置验证 (阶段 3)

```python
config = CARPConfig()
assert config.vocab_size == 512
assert config.ar_depth == 32
print("✅ Config validation passed")
```

**通过标准**: 所有参数符合预期

#### 3. VAE 训练验证 (阶段 8-10)

**指标要求**:
- 重建损失 < 0.01 (收敛)
- Commitment loss 稳定
- Codebook 使用率 > 80%

**验证方法**:
```bash
# 可视化重建质量
python scripts/visualize_vae_reconstruction.py \
    --checkpoint=outputs/carp_vae/checkpoint-50000
```

#### 4. AR 训练验证 (阶段 8-10)

**指标要求**:
- 训练 loss 持续下降
- Mean accuracy > 60%
- Tail accuracy > 50%

**验证方法**:
```bash
# 查看训练曲线
tensorboard --logdir=outputs/carp_ar/tensorboard
```

#### 5. 推理验证 (阶段 4, 10)

```python
policy = CARPPolicy.from_pretrained("outputs/carp_ar/checkpoint-100000")
policy.eval()

batch = {
    "observation.images.image": torch.randn(1, 1, 3, 256, 256),
    "observation.state": torch.randn(1, 1, 10),
    "task_id": torch.tensor([0]),
}

action = policy.select_action(batch)
assert action.shape == (1, 10)  # 正确的输出形状
print("✅ Inference validation passed")
```

**通过标准**:
- 输出形状正确
- 推理速度 < 100ms/step
- 无运行时错误

#### 6. LIBERO 评估验证 (阶段 9)

**最低要求**:
- 环境能正常 reset 和 step
- 策略能运行完整 episode
- 成功率 > 0% (至少有部分任务成功)

**期望目标**:
- Average success rate > 60% (经过充分训练)
- 所有任务都有 > 0% 成功率

### 最终验证清单

#### 功能完整性

- [ ] ✅ CARPConfig 实现所有必需方法
- [ ] ✅ CARPVAEPolicy 实现所有接口
- [ ] ✅ CARPPolicy 实现所有接口
- [ ] ✅ 处理器正确转换数据格式
- [ ] ✅ 工厂函数正确创建模型
- [ ] ✅ 观察编码器正常工作
- [ ] ✅ 两阶段训练流程畅通

#### 训练稳定性

- [ ] ✅ VAE 训练可重复，结果一致 (seed 固定)
- [ ] ✅ AR 训练可重复，结果一致
- [ ] ✅ 损失曲线平滑，无异常波动
- [ ] ✅ checkpoint 保存和加载正确
- [ ] ✅ 支持训练中断和恢复

#### 评估可行性

- [ ] ✅ 可在 LIBERO 环境中评估
- [ ] ✅ 多任务评估正常运行
- [ ] ✅ 评估结果可复现
- [ ] ✅ 支持批量评估
- [ ] ✅ 评估日志完整

#### 代码质量

- [ ] ✅ 通过 ruff 格式检查
- [ ] ✅ 通过 mypy 类型检查
- [ ] ✅ 所有函数有 docstring
- [ ] ✅ 关键部分有注释
- [ ] ✅ 无明显的代码异味

#### 文档完整性

- [ ] ✅ 安装文档完整
- [ ] ✅ 训练文档完整
- [ ] ✅ 评估文档完整
- [ ] ✅ API 参考文档完整
- [ ] ✅ 故障排除文档完整

#### 性能达标

- [ ] ✅ 训练速度 >= CARP 原始实现的 80%
- [ ] ✅ 推理速度 < 100ms/step
- [ ] ✅ GPU 内存使用合理 (< 16GB per GPU)
- [ ] ✅ 最终性能与原始 CARP 接近 (±5%)

---

## 🚀 快速开始指南 (实施后)

### 1. 环境准备

```bash
# 激活环境
conda activate carp_le

# 进入项目目录
cd /inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/Projects/lerobot

# 验证安装
python -c "from lerobot.policies.carp import CARPConfig; print('✅ CARP installed')"
```

### 2. 数据准备

```bash
# 方式 1: 使用现有 LeRobotDataset
# 假设数据已经在 HuggingFace Hub 上
export DATASET_REPO_ID="your/libero_dataset"

# 方式 2: 转换本地数据
# python scripts/convert_to_lerobot_dataset.py --input_dir=data/raw --output_dir=data/lerobot
```

### 3. Stage 1: 训练 VAE

```bash
# 编辑 scripts/train_carp_vae.sh 中的数据路径
# 然后运行
bash scripts/train_carp_vae.sh

# 预计时间: 2-3 小时 (4x GPU, batch_size=256)
# 监控训练: tensorboard --logdir=outputs/carp_vae/tensorboard
```

**预期输出**:
```
Step 10000: loss=0.0234, recon_loss=0.0198, commit_loss=0.0036
Step 20000: loss=0.0156, recon_loss=0.0129, commit_loss=0.0027
Step 30000: loss=0.0098, recon_loss=0.0076, commit_loss=0.0022
Step 40000: loss=0.0067, recon_loss=0.0051, commit_loss=0.0016
Step 50000: loss=0.0045, recon_loss=0.0034, commit_loss=0.0011
✅ VAE training completed
```

### 4. Stage 2: 训练 AR

```bash
# 修改 scripts/train_carp_ar.sh 中的 VAE checkpoint 路径
VAE_CKPT="outputs/carp_vae/checkpoint-50000/model.pth"

# 运行
bash scripts/train_carp_ar.sh

# 预计时间: 6-8 小时 (4x GPU, batch_size=128)
# 监控训练: tensorboard --logdir=outputs/carp_ar/tensorboard
```

**预期输出**:
```
Step 10000: loss=4.567, acc_mean=0.234, acc_tail=0.189
Step 20000: loss=3.892, acc_mean=0.456, acc_tail=0.398
Step 50000: loss=2.345, acc_mean=0.678, acc_tail=0.623
Step 100000: loss=1.234, acc_mean=0.812, acc_tail=0.789
✅ AR training completed
```

### 5. 评估

```bash
# 单任务快速评估
bash scripts/eval_carp_libero.sh

# 多任务完整评估
python scripts/eval_carp_multitask.py \
    --policy_path=outputs/carp_ar/checkpoint-100000 \
    --task_suite=libero_10
```

**预期输出**:
```
=== Evaluating Task 0 ===
Task 0 Success Rate: 84.00%

=== Evaluating Task 1 ===
Task 1 Success Rate: 78.00%

...

=== Summary ===
Average Success Rate: 81.25%
```

---

## 📚 参考资源

### 核心论文和代码

- **CARP 论文**: [arXiv:2412.06782](https://arxiv.org/abs/2412.06782)
- **CARP 项目主页**: https://carp-robot.github.io/
- **CARP 代码**: `carp/multitask/`
- **LeRobot 文档**: https://huggingface.co/docs/lerobot
- **LeRobot GitHub**: https://github.com/huggingface/lerobot

### 相关技术

- **VQ-VAE**: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
- **Autoregressive Models**: [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)
- **LIBERO**: https://libero-project.github.io/
- **Robomimic**: https://robomimic.github.io/

### 依赖库文档

- **PyTorch**: https://pytorch.org/docs/
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers
- **Gymnasium**: https://gymnasium.farama.org/
- **Typed Argument Parser (Tap)**: https://github.com/swansonk14/typed-argument-parser

---

## 💡 关键技术要点总结

### 1. 保留 CARP 核心实现

**原则**: 不修改 MSAT, CFAP 的核心逻辑

**原因**:
- 降低引入 bug 的风险
- 保持与原论文的一致性
- 便于后续维护和更新

**实施**:
- 直接复制核心模块
- 只修改导入路径
- 创建适配层而非修改核心

### 2. 两阶段训练流程

**Stage 1: MSAT**
- 独立训练 VQ-VAE
- 学习多尺度动作表示
- 输出: 预训练的 tokenizer

**Stage 2: CFAP**
- 加载冻结的 VAE
- 训练自回归 Transformer
- 输出: 完整的策略模型

**关键点**:
- VAE 必须充分训练后再训练 AR
- VAE 在 Stage 2 中保持冻结
- 两个阶段使用不同的优化器配置

### 3. 数据流转换

```
LeRobotDataset (Parquet + MP4)
    ↓
PolicyProcessorPipeline (预处理)
    ↓
归一化 + 格式转换
    ↓
CARP 模型输入
    ↓
CARP 模型输出
    ↓
PolicyProcessorPipeline (后处理)
    ↓
Robot/Environment
```

**关键转换**:
- 特征重命名: `image` → `agentview_image`
- 维度调整: (B, T, C, H, W) ↔ CARP 格式
- 归一化: MEAN_STD for state/action, IDENTITY for images

### 4. 观察编码器设计

**输入**: 多模态观察 (图像 + 状态)

**架构**:
```
Images → ResNet18 → [128 per camera]
State  → Identity  → [D]
       ↓
    Concat → [obs_dim]
```

**输出**: 固定维度的特征向量

### 5. 多任务支持

**机制**: 任务嵌入层

```python
task_embed = nn.Embedding(task_num, task_embed_dim)
task_feat = task_embed(task_id)  # (B, task_embed_dim)
obs_feat = torch.cat([obs_feat, task_feat], dim=-1)
```

**推理时**: 需要传递 task_id

### 6. 推理优化

**动作队列**:
```python
_action_queue = deque(maxlen=action_horizon)

# 预测一次
actions = predict_action_chunk()  # (16, 10)

# 逐步执行
for i in range(len(actions)):
    _action_queue.append(actions[i])

# 每步弹出一个
action = _action_queue.popleft()
```

**KV Cache**: 在 AR 推理时启用

---

## 📝 开发日志模板

### 日志格式

```markdown
## 日期: YYYY-MM-DD

### 完成的任务
- [ ] 任务 1
- [ ] 任务 2

### 遇到的问题
1. **问题描述**: ...
   - **原因**: ...
   - **解决方案**: ...

### 下一步计划
- [ ] 任务 3
- [ ] 任务 4

### 备注
- ...
```

### 示例日志

```markdown
## 日期: 2026-03-21

### 完成的任务
- [x] 创建 CARP 核心代码集成计划
- [x] 分析 CARP 和 LeRobot 的架构差异
- [x] 编写详细的实施文档

### 遇到的问题
1. **问题描述**: 依赖版本冲突
   - **原因**: CARP 使用 torch 1.13, LeRobot 使用 torch 2.2
   - **解决方案**: 升级 CARP 代码到 torch 2.2

### 下一步计划
- [ ] 开始阶段 1: 环境准备
- [ ] 测试 CARP 模块导入

### 备注
- 需要与用户确认数据集格式
```

---

## 🎓 总结

本文档详细规划了将 CARP multitask 模型集成到 LeRobot 框架的完整流程，包括：

✅ **12 个详细的实施阶段**，从环境准备到最终交付
✅ **混合方法策略**，保留 CARP 核心实现 + LeRobot 适配层
✅ **全面的风险管理**，识别关键风险并提供应对策略
✅ **明确的验证标准**，确保每个阶段的质量
✅ **完整的时间规划**，预计 12-15 天完成

### 核心优势

1. **快速复现**: 最小化修改，降低风险
2. **保持正确性**: 不改动 CARP 核心算法
3. **符合规范**: 遵循 LeRobot 的代码规范
4. **易于维护**: 清晰的分层架构
5. **可扩展性**: 支持未来的功能扩展

### 下一步行动

1. ✅ 获取用户确认和反馈
2. ✅ 调整计划细节
3. ✅ 开始实施阶段 1

---

**文档维护**: 本文档将随着项目进展持续更新。

**反馈渠道**: 如有问题或建议，请及时沟通。
