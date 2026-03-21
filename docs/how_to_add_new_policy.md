# LeRobot 框架：Policy 集成指南

## 目录

1. [PI05 文件夹结构分析](#pi05-文件夹结构分析)
2. [集成新 Policy 的步骤](#集成新-policy-的步骤)
3. [详细实现指南](#详细实现指南)
4. [注册与工厂模式](#注册与工厂模式)
5. [最佳实践](#最佳实践)

---

## PI05 文件夹结构分析

`pi05` 文件夹包含以下核心文件：

```
pi05/
├── __init__.py              # 模块导出
├── configuration_pi05.py    # 配置类
├── modeling_pi05.py         # 模型实现
├── processor_pi05.py        # 数据处理器
└── README.md               # 文档说明
```

### 1. `configuration_pi05.py` - 配置类

**作用**：定义 PI05 Policy 的所有超参数和配置选项。

**核心功能**：
- 继承自 `PreTrainedConfig`，这是所有 policy 配置的基类
- 使用 `@PreTrainedConfig.register_subclass("pi05")` 装饰器注册配置类型
- 定义模型架构参数（如 `paligemma_variant`, `action_expert_variant`）
- 定义训练参数（如 `optimizer_lr`, `scheduler_warmup_steps`）
- 定义推理参数（如 `num_inference_steps`, `chunk_size`）
- 定义归一化映射 (`normalization_mapping`)

**关键方法**：
```python
@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    # 模型变体配置
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    
    # 动作预测配置
    chunk_size: int = 50  # 预测的动作步数
    n_action_steps: int = 50  # 执行的动作步数
    
    # 必须实现的抽象方法
    def get_optimizer_preset(self) -> AdamWConfig: ...
    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig: ...
    def validate_features(self) -> None: ...
    
    # 必须实现的抽象属性
    @property
    def observation_delta_indices(self) -> None: ...
    @property
    def action_delta_indices(self) -> list: ...
    @property
    def reward_delta_indices(self) -> None: ...
```

---

### 2. `modeling_pi05.py` - 模型实现

**作用**：实现 PI05 Policy 的神经网络模型和推理逻辑。

**核心组件**：

#### 2.1 底层模型类 `PI05Pytorch`
- 封装核心的 PyTorch 模型逻辑
- 实现 flow matching 算法用于动作生成
- 包含 `forward()` 方法用于训练（计算损失）
- 包含 `sample_actions()` 方法用于推理（生成动作）

```python
class PI05Pytorch(nn.Module):
    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        # 初始化 PaliGemma + Expert 模型
        self.paligemma_with_expert = PaliGemmaWithExpertModel(...)
        self.action_in_proj = nn.Linear(...)
        self.action_out_proj = nn.Linear(...)
    
    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None):
        """训练前向传播，计算 MSE 损失"""
        ...
    
    def sample_actions(self, images, img_masks, tokens, masks, ...):
        """推理时生成动作序列"""
        ...
```

#### 2.2 Policy 封装类 `PI05Policy`
- 继承自 `PreTrainedPolicy` 基类
- 必须定义 `config_class` 和 `name` 类属性
- 实现所有抽象方法

```python
class PI05Policy(PreTrainedPolicy):
    config_class = PI05Config  # 关联配置类
    name = "pi05"              # Policy 名称标识符

    def __init__(self, config: PI05Config):
        super().__init__(config)
        self.model = PI05Pytorch(config, ...)
        self.reset()
    
    # 必须实现的抽象方法
    def get_optim_params(self) -> dict: ...
    def reset(self): ...
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]: ...
    def select_action(self, batch: dict[str, Tensor]) -> Tensor: ...
    def predict_action_chunk(self, batch: dict[str, Tensor], ...) -> Tensor: ...
```

---

### 3. `processor_pi05.py` - 数据处理器

**作用**：定义数据预处理和后处理流水线，将原始数据转换为模型输入格式，以及将模型输出转换回原始格式。

**核心功能**：

#### 3.1 自定义处理步骤
```python
@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    预处理状态数据并准备语言 tokenizer 输入
    - 填充状态向量到 max_state_dim
    - 离散化状态到 256 个 bins
    - 构建完整的文本提示
    """
    def __call__(self, transition: EnvTransition) -> EnvTransition: ...
    def transform_features(self, features: ...) -> ...: ...
```

#### 3.2 预处理和后处理流水线工厂函数
```python
def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """
    创建预处理器和后处理器流水线
    
    预处理流水线步骤：
    1. RenameObservationsProcessorStep - 重命名特征
    2. AddBatchDimensionProcessorStep - 添加批次维度
    3. NormalizerProcessorStep - 归一化（必须在 tokenizer 之前！）
    4. Pi05PrepareStateTokenizerProcessorStep - 准备状态和任务文本
    5. TokenizerProcessorStep - 文本 tokenization
    6. DeviceProcessorStep - 移动到目标设备
    
    后处理流水线步骤：
    1. UnnormalizerProcessorStep - 反归一化
    2. DeviceProcessorStep - 移动到 CPU
    """
```

---

### 4. `__init__.py` - 模块导出

```python
from .configuration_pi05 import PI05Config
from .modeling_pi05 import PI05Policy
from .processor_pi05 import make_pi05_pre_post_processors

__all__ = ["PI05Config", "PI05Policy", "make_pi05_pre_post_processors"]
```

---

## 集成新 Policy 的步骤

### 第一步：创建 Policy 文件夹结构

```bash
src/lerobot/policies/my_policy/
├── __init__.py
├── configuration_my_policy.py
├── modeling_my_policy.py
├── processor_my_policy.py
└── README.md  # (可选) 文档说明
```

### 第二步：实现配置类 (`configuration_my_policy.py`)

```python
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig

@PreTrainedConfig.register_subclass("my_policy")
@dataclass
class MyPolicyConfig(PreTrainedConfig):
    """你的 Policy 配置类"""
    
    # ===== 自定义参数 =====
    hidden_dim: int = 256
    num_layers: int = 4
    # ... 其他参数
    
    # ===== 归一化映射 =====
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    
    # ===== 优化器参数 =====
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 0.01
    
    def __post_init__(self):
        super().__post_init__()
        # 自定义验证逻辑
    
    def validate_features(self) -> None:
        """验证和设置输入/输出特征"""
        # 确保必需的特征存在
        pass
    
    def get_optimizer_preset(self) -> AdamWConfig:
        """返回优化器配置"""
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )
    
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """返回学习率调度器配置（可选）"""
        return None  # 或返回具体的调度器配置
    
    @property
    def observation_delta_indices(self) -> list | None:
        return None
    
    @property
    def action_delta_indices(self) -> list | None:
        return None
    
    @property
    def reward_delta_indices(self) -> list | None:
        return None
```

### 第三步：实现模型类 (`modeling_my_policy.py`)

```python
import torch
from torch import Tensor, nn
from collections import deque

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.my_policy.configuration_my_policy import MyPolicyConfig
from lerobot.utils.constants import ACTION


class MyPolicyNetwork(nn.Module):
    """核心神经网络实现"""
    
    def __init__(self, config: MyPolicyConfig):
        super().__init__()
        self.config = config
        # 定义网络层
        self.encoder = ...
        self.decoder = ...
    
    def forward(self, observations, actions=None):
        """
        训练时：返回损失
        推理时：返回预测的动作
        """
        pass


class MyPolicy(PreTrainedPolicy):
    """LeRobot Policy 封装类"""
    
    # 必须定义这两个类属性
    config_class = MyPolicyConfig
    name = "my_policy"
    
    def __init__(self, config: MyPolicyConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        
        # 初始化网络
        self.model = MyPolicyNetwork(config)
        self.model.to(config.device)
        
        self.reset()
    
    def get_optim_params(self) -> dict:
        """返回需要优化的参数"""
        return self.parameters()
    
    def reset(self):
        """重置内部状态（环境重置时调用）"""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
    
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        训练前向传播
        
        Args:
            batch: 包含观测和动作的字典
            
        Returns:
            loss: 标量损失张量
            loss_dict: 包含日志信息的字典
        """
        # 准备输入
        observations = self._prepare_observations(batch)
        actions = batch[ACTION]
        
        # 计算损失
        loss = self.model(observations, actions)
        
        loss_dict = {
            "loss": loss.item(),
        }
        
        return loss, loss_dict
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """
        预测一个动作块
        
        Returns:
            actions: shape (batch_size, chunk_size, action_dim)
        """
        self.eval()
        observations = self._prepare_observations(batch)
        actions = self.model.predict(observations)
        return actions
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """
        选择单个动作（用于实时控制）
        
        实现动作队列逻辑以支持 n_action_steps > 1
        """
        self.eval()
        
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, :self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        
        return self._action_queue.popleft()
    
    def _prepare_observations(self, batch: dict[str, Tensor]):
        """准备观测数据"""
        # 实现观测预处理逻辑
        pass
```

### 第四步：实现处理器 (`processor_my_policy.py`)

```python
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.my_policy.configuration_my_policy import MyPolicyConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register(name="my_policy_custom_processor_step")
@dataclass
class MyPolicyCustomProcessorStep(ProcessorStep):
    """
    自定义处理步骤（如果需要）
    """
    
    some_param: int = 32
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """处理 transition 数据"""
        transition = transition.copy()
        # 实现自定义处理逻辑
        return transition
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """声明特征变换（如果有）"""
        return features


def make_my_policy_pre_post_processors(
    config: MyPolicyConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    创建预处理器和后处理器流水线
    """
    
    # ===== 预处理步骤 =====
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        # 添加自定义步骤（如果需要）
        # MyPolicyCustomProcessorStep(some_param=config.some_param),
        DeviceProcessorStep(device=config.device),
    ]
    
    # ===== 后处理步骤 =====
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
```

### 第五步：创建模块导出 (`__init__.py`)

```python
from .configuration_my_policy import MyPolicyConfig
from .modeling_my_policy import MyPolicy
from .processor_my_policy import make_my_policy_pre_post_processors

__all__ = ["MyPolicyConfig", "MyPolicy", "make_my_policy_pre_post_processors"]
```

---

## 注册与工厂模式

### 第六步：在 `policies/__init__.py` 中注册配置

编辑 `src/lerobot/policies/__init__.py`：

```python
# 添加导入
from .my_policy.configuration_my_policy import MyPolicyConfig as MyPolicyConfig

# 更新 __all__
__all__ = [
    # ... 现有的
    "MyPolicyConfig",
]
```

### 第七步：在 `factory.py` 中注册 Policy

编辑 `src/lerobot/policies/factory.py`：

#### 7.1 在 `get_policy_class` 函数中添加

```python
def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    # ... 现有代码
    elif name == "my_policy":
        from lerobot.policies.my_policy.modeling_my_policy import MyPolicy
        return MyPolicy
    # ...
```

#### 7.2 在 `make_policy_config` 函数中添加

```python
def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    # ... 现有代码
    elif policy_type == "my_policy":
        from lerobot.policies.my_policy.configuration_my_policy import MyPolicyConfig
        return MyPolicyConfig(**kwargs)
    # ...
```

#### 7.3 在 `make_pre_post_processors` 函数中添加

```python
def make_pre_post_processors(...):
    # ... 现有代码
    elif isinstance(policy_cfg, MyPolicyConfig):
        from lerobot.policies.my_policy.processor_my_policy import make_my_policy_pre_post_processors
        processors = make_my_policy_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )
    # ...
```

#### 7.4 添加必要的导入

```python
from lerobot.policies.my_policy.configuration_my_policy import MyPolicyConfig
```

---

## 最佳实践

### 1. 配置类设计

- 使用 `@dataclass` 和 `field(default_factory=...)` 处理可变默认值
- 在 `__post_init__` 中进行参数验证
- 提供合理的默认值
- 使用类型提示

### 2. 模型实现

- 将核心网络逻辑放在独立的 `nn.Module` 中
- Policy 类主要负责接口适配和状态管理
- 支持 `torch.compile` 和梯度检查点（可选）
- 实现 `from_pretrained` 方法处理预训练权重加载（如果有特殊需求）

### 3. 处理器设计

- 每个处理步骤应该是原子操作
- 使用 `ProcessorStepRegistry.register` 注册自定义步骤
- 预处理器和后处理器应该是可逆的（归一化 <-> 反归一化）
- 注意处理步骤的顺序

### 4. 测试

创建测试文件 `tests/policies/test_my_policy.py`：

```python
import pytest
import torch

from lerobot.policies.my_policy import MyPolicy, MyPolicyConfig


def test_my_policy_config():
    config = MyPolicyConfig()
    assert config.hidden_dim == 256


def test_my_policy_forward():
    config = MyPolicyConfig(device="cpu")
    policy = MyPolicy(config)
    
    # 创建模拟输入
    batch = {
        "observation.state": torch.randn(2, 10),
        "action": torch.randn(2, 50, 6),
    }
    
    loss, loss_dict = policy.forward(batch)
    assert loss.requires_grad
```

### 5. 文档

- 在 `README.md` 中说明 policy 的原理和用法
- 添加使用示例和参考论文
- 说明特殊的配置选项

---

## 完整文件清单

集成新 policy 需要修改/创建的文件：

| 文件路径 | 操作 | 说明 |
|---------|------|------|
| `src/lerobot/policies/my_policy/__init__.py` | 创建 | 模块导出 |
| `src/lerobot/policies/my_policy/configuration_my_policy.py` | 创建 | 配置类 |
| `src/lerobot/policies/my_policy/modeling_my_policy.py` | 创建 | 模型实现 |
| `src/lerobot/policies/my_policy/processor_my_policy.py` | 创建 | 数据处理器 |
| `src/lerobot/policies/__init__.py` | 修改 | 添加配置导入 |
| `src/lerobot/policies/factory.py` | 修改 | 注册工厂方法 |
| `tests/policies/test_my_policy.py` | 创建 | 单元测试 |
| `docs/source/policy_my_policy_README.md` | 创建 | 文档（可选） |

---

## 参考现有实现

建议参考以下现有 policy 实现：

- **ACT** (`policies/act/`): 简单清晰的实现，适合入门参考
- **Diffusion** (`policies/diffusion/`): 扩散模型实现
- **PI05** (`policies/pi05/`): 大型 VLM 模型，复杂的预处理流程
- **SmolVLA** (`policies/smolvla/`): 另一个 VLA 模型实现

每个实现都遵循相同的模式：配置类 + 模型类 + 处理器，这是 LeRobot 框架的核心设计范式。
