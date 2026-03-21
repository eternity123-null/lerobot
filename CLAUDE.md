# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**LeRobot** 是 Hugging Face 开发的 PyTorch 机器人学习库,旨在降低机器人学习的门槛。项目提供:
- **统一的硬件无关 Robot 接口**:支持多种机器人硬件(SO100, Koch, Reachy2, Unitree G1等)
- **标准化的 LeRobotDataset 格式**:Parquet + MP4/images,与 HF Hub 无缝集成
- **多种最先进的策略实现**:模仿学习(ACT, Diffusion, VQ-BeT)、强化学习(TDMPC, SAC)、VLA模型(Pi0/Pi0.5, Groot, SmolVLA, XVLA)
- **完整的端到端工作流**:数据收集 → 训练 → 评估 → 部署
- **可扩展的架构**:基于工厂模式和注册表,易于扩展新策略、机器人和环境


## 开发任务
   我想利用lerobot框架复现CARP项目(@carp/)multitask部分的模型,便于训练后在libero环境中进行测试。你需要在深度理解carp的技术原理思路和lerobot框架的基础上进行实现。carp文件夹下有对应的论文，可以在必要时查阅。

## 核心命令

### 训练与评估

```bash
# 训练策略
lerobot-train --policy=act --dataset.repo_id=lerobot/aloha_mobile_cabinet

# 评估策略
lerobot-eval --policy.path=lerobot/pi0_libero_finetuned --env.type=libero --env.task=libero_object --eval.n_episodes=10

# 可视化数据集
lerobot-dataset-viz --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

### 机器人操作

```bash
# 校准机器人
lerobot-calibrate --robot.path=lerobot/configs/robot/koch.yaml

# 远程操作(数据收集)
lerobot-teleoperate --robot.path=lerobot/configs/robot/koch.yaml

# 录制数据
lerobot-record --robot.path=lerobot/configs/robot/koch.yaml --repo-id=my_username/my_dataset

# 回放数据
lerobot-replay --robot.path=lerobot/configs/robot/koch.yaml --repo-id=my_username/my_dataset

# 查找相机/端口
lerobot-find-cameras
lerobot-find-port
```

### 开发命令

```bash
# 安装开发依赖
pip install -e ".[dev,test]"

# 运行代码质量检查
pre-commit run --all-files

# 运行测试(需要 git-lfs)
git lfs install && git lfs pull
pytest -sv ./tests

# 运行单个测试文件
pytest -sv tests/policies/test_policies.py

# 运行特定测试
pytest -sv tests/policies/test_policies.py::test_policy_forward
```

### 开发环境
 在 carp_le 的conda环境中完成测试，缺失package安装等工作。注意！carp_le已经是配置好了lerobot框架所需的几乎所有package了，但是没有安装carp的包。为了防止carp的包扰乱lerobot环境，只有在需要用到的情况下再安装包，不要直接安装environment.yaml的所有包

## 核心架构设计

### 1. 模块组织

```
src/lerobot/
├── policies/           # 策略实现 (ACT, Diffusion, VQ-BeT, Pi0/Pi0.5, Groot, SmolVLA, XVLA等)
│   ├── factory.py      # Policy 工厂:get_policy_class(), make_policy(), make_pre_post_processors()
│   ├── pretrained.py   # PreTrainedPolicy 基类:from_pretrained(), save_pretrained()
│   └── {policy_name}/  # 每个策略包含三个核心文件:
│       ├── configuration_{policy}.py  # 配置类 (继承 PreTrainedConfig, 使用 @register_subclass)
│       ├── modeling_{policy}.py       # 模型实现 (继承 PreTrainedPolicy)
│       └── processor_{policy}.py      # make_{policy}_pre_post_processors() 工厂函数
│
├── datasets/           # 数据集管理
│   ├── lerobot_dataset.py     # LeRobotDataset 核心类 (v3.0)
│   ├── compute_stats.py       # 数据统计计算 (用于归一化)
│   ├── sampler.py             # EpisodeAwareSampler (episode 边界感知采样)
│   └── factory.py             # make_dataset() 工厂函数
│
├── configs/            # 配置系统 (基于 draccus)
│   ├── policies.py     # PreTrainedConfig 抽象基类 + ChoiceRegistry
│   ├── train.py        # TrainPipelineConfig (训练配置)
│   ├── types.py        # FeatureType, NormalizationMode, PolicyFeature 类型定义
│   └── parser.py       # CLI 参数解析 (@parser.wrap() 装饰器)
│
├── processor/          # 数据处理流水线
│   ├── pipeline.py     # PolicyProcessorPipeline 和 ProcessorStep 抽象类
│   ├── core.py         # EnvTransition, PolicyAction, RobotAction 类型定义
│   ├── converters.py   # transition ↔ batch, policy_action ↔ transition 转换器
│   └── steps/          # 各种 ProcessorStep 实现 (归一化、设备转换、tokenizer等)
│       ├── normalize.py      # NormalizerProcessorStep / UnnormalizerProcessorStep
│       ├── device.py         # DeviceProcessorStep
│       └── ...
│
├── robots/             # 机器人硬件抽象层
│   ├── robot.py        # Robot 抽象基类:connect(), get_observation(), send_action()
│   └── {robot_name}/   # 具体机器人实现 (SO100, Koch, Reachy2, Unitree G1等)
│
├── cameras/            # 相机接口 (OpenCV, RealSense, ZMQ等)
├── motors/             # 电机控制 (Dynamixel, Feetech)
├── teleoperators/      # 远程操作设备 (Gamepad, Keyboard, Phone等)
│
├── envs/               # 仿真环境 (基于 Gymnasium)
│   ├── factory.py      # make_env(), make_env_pre_post_processors()
│   ├── libero.py       # LIBERO 环境包装器
│   └── metaworld.py    # MetaWorld 环境包装器
│
├── optim/              # 优化和调度
│   ├── optimizers.py   # OptimizerConfig, make_optimizer()
│   └── schedulers.py   # LRSchedulerConfig, make_scheduler()
│
├── async_inference/    # 异步推理系统 (gRPC 服务器-客户端架构)
│   ├── policy_server.py   # GPU 上运行的策略推理服务器
│   └── robot_client.py    # 机器人端客户端
│
├── model/              # 通用模型组件 (vision encoders, transformers等)
├── rl/                 # 强化学习组件 (actor-learner 架构)
├── utils/              # 工具函数
└── scripts/            # CLI 入口点
    ├── lerobot_train.py      # 训练主脚本
    ├── lerobot_eval.py       # 评估脚本
    ├── lerobot_record.py     # 数据记录
    └── ...
```

### 2. 配置系统架构 (基于 draccus)

LeRobot 使用 **draccus** 库实现类型安全的配置系统,支持 dataclass、YAML 和 CLI 参数的无缝集成。

#### PreTrainedConfig (策略配置基类)

位置:`src/lerobot/configs/policies.py`

```python
@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    """所有策略配置的抽象基类"""

    # 必须实现的抽象方法
    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig: ...

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None: ...

    @abc.abstractmethod
    def validate_features(self) -> None: ...

    # 必须实现的抽象属性
    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list | None: ...

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list | None: ...

    # 核心字段
    n_obs_steps: int = 1  # 观测历史长度
    input_features: dict[str, PolicyFeature] | None = None  # 从数据集推断
    output_features: dict[str, PolicyFeature] | None = None
    device: str | None = None  # "cuda", "cpu", "mps"
    use_amp: bool = False  # 自动混合精度
    use_peft: bool = False  # 参数高效微调
```

**关键特性:**
- **ChoiceRegistry**:使用 `@PreTrainedConfig.register_subclass("policy_name")` 注册子类
- **HubMixin**:支持 `from_pretrained()` 和 `save_pretrained()` 方法
- **自动设备检测**:`__post_init__` 中自动选择可用设备

#### TrainPipelineConfig (训练配置)

位置:`src/lerobot/configs/train.py`

```python
@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig  # 数据集配置
    env: EnvConfig | None = None  # 评估环境配置
    policy: PreTrainedConfig | None = None  # 策略配置
    output_dir: Path | None = None
    resume: bool = False
    seed: int = 1000
    batch_size: int = 8
    steps: int = 100_000
    eval_freq: int = 20_000
    log_freq: int = 200
    save_freq: int = 20_000
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    wandb: WandBConfig = field(default_factory=WandBConfig)
    peft: PeftConfig | None = None

    # RA-BC (Reward-Aligned Behavior Cloning) 参数
    use_rabc: bool = False  # 基于奖励的样本加权
    rabc_progress_path: str | None = None  # SARM 进度文件路径
```

**配置加载流程:**
1. **从 CLI 加载**: `lerobot-train --policy=act --dataset.repo_id=...`
2. **从预训练模型加载**: `--policy.path=lerobot/act_aloha` (自动加载 config.json)
3. **从 checkpoint 恢复**: `--resume --config_path=outputs/.../train_config.json`

#### PolicyFeature 和 FeatureType

位置:`src/lerobot/configs/types.py`

```python
class FeatureType(str, Enum):
    STATE = "STATE"        # 机器人状态 (关节位置、速度等)
    VISUAL = "VISUAL"      # 视觉观测 (相机图像)
    ENV = "ENV"            # 环境状态
    ACTION = "ACTION"      # 动作
    REWARD = "REWARD"      # 奖励
    LANGUAGE = "LANGUAGE"  # 语言指令

@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]  # 不包含 batch 和时间维度

class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"       # [-1, 1] 归一化
    MEAN_STD = "MEAN_STD"     # 零均值单位方差
    IDENTITY = "IDENTITY"      # 不归一化
    QUANTILES = "QUANTILES"    # 分位数归一化
```

### 3. 处理器架构 (Processor Pipeline)

位置:`src/lerobot/processor/`

处理器系统负责在 **Robot/Env ↔ Policy** 之间进行数据转换。

#### 核心概念

```python
# 数据类型定义 (processor/core.py)
PolicyAction: TypeAlias = torch.Tensor  # Policy 输出的动作张量
RobotAction: TypeAlias = dict[str, Any]  # Robot 期望的动作字典
EnvAction: TypeAlias = np.ndarray        # Env 期望的动作数组
RobotObservation: TypeAlias = dict[str, Any]  # Robot/Env 提供的观测

EnvTransition = TypedDict(
    "EnvTransition",
    {
        "observation": RobotObservation | None,
        "action": PolicyAction | RobotAction | EnvAction | None,
        "reward": float | torch.Tensor | None,
        "done": bool | torch.Tensor | None,
        "truncated": bool | torch.Tensor | None,
        "info": dict[str, Any] | None,
        "complementary_data": dict[str, Any] | None,
    },
)
```

#### ProcessorStep 抽象类

```python
class ProcessorStep(ABC):
    """数据处理步骤的抽象基类"""

    @abstractmethod
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """对 transition 进行转换"""
        pass

    @abstractmethod
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """声明此步骤如何改变特征形状/类型"""
        pass

    # 可选:有状态的 ProcessorStep 可实现
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict): ...
```

#### ProcessorStepRegistry (注册表模式)

```python
@ProcessorStepRegistry.register(name="normalize")
@dataclass
class NormalizerProcessorStep(ProcessorStep):
    """归一化处理步骤"""
    features: dict[str, PolicyFeature]
    norm_map: dict[str, NormalizationMode]
    stats: dict[str, dict[str, torch.Tensor]]  # {"mean": ..., "std": ...}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # 对观测和动作进行归一化
        ...
```

**注册和获取:**
- `ProcessorStepRegistry.register(name="custom_step")` - 注册自定义步骤
- `ProcessorStepRegistry.get(name)` - 通过名称获取步骤类
- `ProcessorStepRegistry.list()` - 列出所有已注册步骤

#### PolicyProcessorPipeline (流水线)

```python
class PolicyProcessorPipeline(Generic[TInput, TOutput], HubMixin):
    """处理器流水线:按顺序执行多个 ProcessorStep"""

    def __init__(
        self,
        steps: list[ProcessorStep],
        name: str,
        to_transition: Callable[[TInput], EnvTransition],
        to_output: Callable[[EnvTransition], TOutput],
    ):
        self.steps = steps
        self.name = name
        self.to_transition = to_transition  # 输入 → EnvTransition
        self.to_output = to_output          # EnvTransition → 输出

    def __call__(self, input: TInput) -> TOutput:
        transition = self.to_transition(input)
        for step in self.steps:
            transition = step(transition)
        return self.to_output(transition)

    # Hub 集成
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, ...): ...
    def save_pretrained(self, save_directory): ...
```

#### 预处理器和后处理器工厂

每个策略都有一个 `make_{policy}_pre_post_processors()` 函数:

```python
def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],  # 预处理器
    PolicyProcessorPipeline[PolicyAction, PolicyAction],       # 后处理器
]:
    """
    预处理器流水线 (Robot → Policy):
    1. RenameObservationsProcessorStep - 重命名特征键
    2. AddBatchDimensionProcessorStep - 添加 batch 维度
    3. NormalizerProcessorStep - 归一化观测和动作
    4. DeviceProcessorStep - 移动到目标设备 (GPU/CPU)

    后处理器流水线 (Policy → Robot):
    1. UnnormalizerProcessorStep - 反归一化动作
    2. DeviceProcessorStep - 移动到 CPU
    """
    input_steps = [
        RenameObservationsProcessorStep(...),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps = [
        UnnormalizerProcessorStep(...),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline(steps=input_steps, ...),
        PolicyProcessorPipeline(steps=output_steps, ...),
    )
```

**关键约定:**
- **归一化必须在 tokenizer 之前!**(对于 VLA 模型)
- 预处理器输入: `dict[str, Any]` (来自 Robot/Env)
- 预处理器输出: `dict[str, torch.Tensor]` (送入 Policy)
- 后处理器输入: `PolicyAction` (Policy 输出)
- 后处理器输出: `PolicyAction` (送回 Robot/Env)

### 4. Policy 架构模式 (标准三组件)

每个策略必须遵循标准的三组件模式:

#### 组件 1: Configuration (`configuration_{policy}.py`)

```python
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig

@PreTrainedConfig.register_subclass("act")  # 注册到 ChoiceRegistry
@dataclass
class ACTConfig(PreTrainedConfig):
    """ACT Policy 配置"""

    # 模型架构参数
    chunk_size: int = 100  # 动作块大小
    n_action_steps: int = 100  # 执行步数
    dim_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    use_vae: bool = True

    # 归一化映射 (关键!)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,  # 图像保持原样
            "STATE": NormalizationMode.MEAN_STD,   # 状态标准化
            "ACTION": NormalizationMode.MEAN_STD,  # 动作标准化
        }
    )

    # 必须实现的方法
    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=1e-4, weight_decay=1e-4, betas=[0.95, 0.999])

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None  # 或返回具体的调度器配置

    def validate_features(self) -> None:
        """验证输入/输出特征的有效性"""
        # 检查必需的特征是否存在
        assert self.input_features is not None
        assert ACTION in self.output_features

    # Delta indices (用于 temporal difference)
    @property
    def observation_delta_indices(self) -> None:
        return None  # 不使用 delta observations

    @property
    def action_delta_indices(self) -> list:
        return [[0, chunk_size]]  # 使用 delta actions
```

#### 组件 2: Modeling (`modeling_{policy}.py`)

```python
from torch import nn, Tensor
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.act.configuration_act import ACTConfig

class ACTModel(nn.Module):
    """核心神经网络实现"""
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.encoder = ...  # Transformer encoder
        self.decoder = ...  # Transformer decoder
        self.vae = ...      # VAE (如果 use_vae=True)

    def forward(self, obs_images, obs_state, actions=None):
        """训练前向传播"""
        # 编码观测
        encoded = self.encoder(obs_images, obs_state)
        # 解码动作
        predicted_actions = self.decoder(encoded, actions)
        return predicted_actions


class ACTPolicy(PreTrainedPolicy):
    """LeRobot Policy 封装类"""

    # 必须定义的类属性
    config_class = ACTConfig
    name = "act"

    def __init__(self, config: ACTConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = ACTModel(config)
        self.model.to(config.device)
        self.reset()

    def get_optim_params(self) -> dict:
        """返回需要优化的参数"""
        return self.parameters()

    def reset(self):
        """重置内部状态 (环境重置时调用)"""
        from collections import deque
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        训练前向传播

        Args:
            batch: {
                "observation.images.{cam}": (B, T, C, H, W),
                "observation.state": (B, T, D),
                "action": (B, T, A),
            }

        Returns:
            loss: 标量损失
            loss_dict: 日志字典 {"loss": float, ...}
        """
        # 提取观测和动作
        obs_images = self._prepare_images(batch)
        obs_state = batch["observation.state"]
        actions = batch["action"]

        # 计算损失
        predicted_actions = self.model(obs_images, obs_state, actions)
        loss = F.mse_loss(predicted_actions, actions)

        loss_dict = {"loss": loss.item()}
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """
        预测一个动作块

        Returns:
            actions: (B, chunk_size, action_dim)
        """
        self.eval()
        obs_images = self._prepare_images(batch)
        obs_state = batch["observation.state"]
        actions = self.model(obs_images, obs_state)  # (B, chunk_size, A)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """
        选择单个动作 (用于实时控制)

        实现动作队列以支持 temporal action chunking
        """
        self.eval()
        if len(self._action_queue) == 0:
            # 预测一个新的 action chunk
            actions = self.predict_action_chunk(batch)  # (1, chunk_size, A)
            actions = actions[:, :self.config.n_action_steps]  # 只取 n_action_steps
            self._action_queue.extend(actions.squeeze(0))

        # 从队列中弹出一个动作
        return self._action_queue.popleft()
```

**关键方法说明:**
- **`forward()`**: 训练时调用,计算损失
- **`select_action()`**: 实时控制时调用,返回单步动作
- **`predict_action_chunk()`**: 批量预测,返回 chunk 动作
- **`reset()`**: 环境重置时清理内部状态
- **`get_optim_params()`**: 返回可优化参数

#### 组件 3: Processor (`processor_{policy}.py`)

```python
from lerobot.processor import (
    PolicyProcessorPipeline,
    ProcessorStep,
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
    DeviceProcessorStep,
    AddBatchDimensionProcessorStep,
    RenameObservationsProcessorStep,
)

def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """创建 ACT 的预处理器和后处理器"""

    # 预处理器步骤
    input_steps: list[ProcessorStep] = [
        # 1. 重命名观测键 (例如: "left" -> "observation.images.left")
        RenameObservationsProcessorStep(rename_map={}),

        # 2. 添加 batch 维度 (如果是单个样本)
        AddBatchDimensionProcessorStep(),

        # 3. 归一化 (使用数据集统计)
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),

        # 4. 移动到目标设备
        DeviceProcessorStep(device=config.device),
    ]

    # 后处理器步骤
    output_steps: list[ProcessorStep] = [
        # 1. 反归一化动作
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),

        # 2. 移动到 CPU
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline(
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        ),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
```

**处理器步骤顺序非常重要!** 对于 VLA 模型:
1. 归一化必须在 tokenizer 之前
2. 设备转换通常放在最后

### 5. 工厂模式 (Factory Pattern)

LeRobot 使用工厂函数来解耦组件创建和使用。

#### Policy 工厂 (`policies/factory.py`)

```python
# 1. 获取 Policy 类
def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """通过名称获取 Policy 类 (动态导入)"""
    if name == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        return ACTPolicy
    elif name == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        return DiffusionPolicy
    # ...
    else:
        # 支持第三方插件
        return _get_policy_cls_from_policy_name(name)

# 2. 创建 Config 实例
def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """创建 Policy 配置"""
    if policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    # ...

# 3. 创建预处理器和后处理器
def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """创建或加载处理器流水线"""

    if pretrained_path:
        # 从预训练路径加载
        return (
            PolicyProcessorPipeline.from_pretrained(
                pretrained_path,
                config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            ),
            PolicyProcessorPipeline.from_pretrained(
                pretrained_path,
                config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            ),
        )

    # 根据配置类型创建新的处理器
    if isinstance(policy_cfg, ACTConfig):
        from lerobot.policies.act.processor_act import make_act_pre_post_processors
        return make_act_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )
    # ...

# 4. 创建 Policy 实例
def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    rename_map: dict[str, str] | None = None,
) -> PreTrainedPolicy:
    """创建 Policy 实例 (从头或从预训练)"""

    # 从数据集或环境推断特征
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
    else:
        features = env_to_policy_features(env_cfg)

    cfg.input_features = {key: ft for key, ft in features.items() if ft.type != FeatureType.ACTION}
    cfg.output_features = {key: ft for key, ft in features.items() if ft.type == FeatureType.ACTION}

    policy_cls = get_policy_class(cfg.type)

    if cfg.pretrained_path:
        # 加载预训练模型
        policy = policy_cls.from_pretrained(cfg.pretrained_path, config=cfg)
    else:
        # 从头创建
        policy = policy_cls(config=cfg)

    policy.to(cfg.device)
    return policy
```

**工厂模式优势:**
- **延迟导入**:只导入需要的策略,加快启动速度
- **解耦**:使用者不需要知道具体的类名和导入路径
- **可扩展**:支持第三方插件通过 ChoiceRegistry 注册

## 核心命令

### 训练与评估

```bash
# 训练策略 (基本)
lerobot-train --policy=act --dataset.repo_id=lerobot/aloha_mobile_cabinet

# 训练策略 (完整配置)
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --batch_size=32 \
  --steps=100000 \
  --eval_freq=5000 \
  --save_freq=5000 \
  --output_dir=outputs/my_training \
  --wandb.enable=true \
  --wandb.project=lerobot

# 从预训练模型微调
lerobot-train \
  --policy.path=lerobot/act_aloha \
  --dataset.repo_id=my_username/my_dataset \
  --optimizer.lr=1e-5

# 恢复训练
lerobot-train \
  --resume \
  --config_path=outputs/my_training/train_config.json

# 评估策略
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10

# 可视化数据集
lerobot-dataset-viz --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

### 机器人操作

```bash
# 校准机器人
lerobot-calibrate --robot.path=lerobot/configs/robot/koch.yaml

# 远程操作 (数据收集)
lerobot-teleoperate --robot.path=lerobot/configs/robot/koch.yaml

# 录制数据
lerobot-record \
  --robot.path=lerobot/configs/robot/koch.yaml \
  --repo-id=my_username/my_dataset \
  --num-episodes=50 \
  --fps=30

# 回放数据
lerobot-replay \
  --robot.path=lerobot/configs/robot/koch.yaml \
  --repo-id=my_username/my_dataset \
  --episode=0

# 查找相机/端口
lerobot-find-cameras
lerobot-find-port

# 查找关节限位
lerobot-find-joint-limits --robot.path=...
```

### 开发命令

```bash
# 安装开发依赖
pip install -e ".[dev,test]"

# 运行代码质量检查
pre-commit run --all-files

# 运行测试 (需要 git-lfs)
git lfs install && git lfs pull
pytest -sv ./tests

# 运行单个测试文件
pytest -sv tests/policies/test_policies.py

# 运行特定测试
pytest -sv tests/policies/test_policies.py::test_act_policy

# 运行带覆盖率的测试
pytest --cov=lerobot --cov-report=html tests/

# 类型检查
mypy src/lerobot/configs/
```

## 关键设计模式和约定

### 1. 注册表模式 (Registry Pattern)

LeRobot 广泛使用注册表模式来实现可扩展性:

```python
# PreTrainedConfig 使用 ChoiceRegistry (来自 draccus)
@PreTrainedConfig.register_subclass("my_policy")
class MyPolicyConfig(PreTrainedConfig):
    ...

# ProcessorStep 使用自定义 ProcessorStepRegistry
@ProcessorStepRegistry.register(name="my_step")
class MyProcessorStep(ProcessorStep):
    ...
```

**优点:**
- 支持第三方插件
- 延迟导入,加快启动速度
- 序列化和反序列化时通过名称查找类

### 2. 工厂模式 (Factory Pattern)

所有主要组件都通过工厂函数创建:

```python
# Policy 工厂
policy = make_policy(cfg, ds_meta=dataset.meta)

# Processor 工厂
preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=stats)

# Dataset 工厂
dataset = make_dataset(dataset_cfg)

# Env 工厂
env = make_env(env_cfg)

# Optimizer 工厂
optimizer, scheduler = make_optimizer_and_scheduler(policy, policy_cfg, ...)
```

**优点:**
- 统一的创建接口
- 自动处理复杂的初始化逻辑
- 易于测试和替换实现

### 3. Hub 集成 (HubMixin)

所有配置和模型都集成了 Hugging Face Hub:

```python
# 保存到 Hub
policy.push_to_hub("my_username/my_policy")

# 从 Hub 加载
policy = ACTPolicy.from_pretrained("my_username/my_policy")

# 配置也支持
config = ACTConfig.from_pretrained("my_username/my_policy")
```

### 4. 数据类型转换链

```
Robot/Env → RobotObservation (dict[str, Any])
    ↓ preprocessor
Policy Input (dict[str, torch.Tensor])
    ↓ policy.forward()
PolicyAction (torch.Tensor)
    ↓ postprocessor
RobotAction (dict[str, Any]) → Robot/Env
```

**关键转换器** (`processor/converters.py`):
- `batch_to_transition(batch)` → `EnvTransition`
- `transition_to_batch(transition)` → `batch`
- `policy_action_to_transition(action)` → `EnvTransition`
- `transition_to_policy_action(transition)` → `PolicyAction`

### 5. 命名约定

**文件和模块:**
- `configuration_{policy}.py` - 配置类
- `modeling_{policy}.py` - 模型实现
- `processor_{policy}.py` - 处理器工厂
- `test_{module}.py` - 测试文件

**类名:**
- `{Policy}Config` - 配置类 (例如: `ACTConfig`)
- `{Policy}Policy` - 策略类 (例如: `ACTPolicy`)
- `{Module}ProcessorStep` - 处理器步骤 (例如: `NormalizerProcessorStep`)

**常量** (`utils/constants.py`):
- `OBS_STATE = "observation.state"`
- `OBS_IMAGES = "observation.images"`
- `ACTION = "action"`
- `REWARD = "next.reward"`

### 6. 张量形状约定

```python
# 观测
observation.state: (B, T, D)  # batch, time, state_dim
observation.images.{cam}: (B, T, C, H, W)  # batch, time, channels, height, width

# 动作
action: (B, T, A)  # batch, horizon, action_dim
# 注意: horizon 可能 != 1 (action chunking)

# Policy 输入 (训练时)
batch = {
    "observation.state": (B, n_obs_steps, D),
    "observation.images.top": (B, n_obs_steps, C, H, W),
    "action": (B, chunk_size, A),
}

# Policy 输出
predicted_actions: (B, chunk_size, A)
```

**重要:**
- `B` = batch size
- `T` = 时间步数 (n_obs_steps 或 chunk_size)
- `D` = 状态维度
- `C` = 通道数 (通常为 3)
- `H, W` = 图像高度和宽度
- `A` = 动作维度

## 开发规范

### 代码风格

项目使用 **ruff** 进行代码检查和格式化:
- 使用 `pre-commit` 自动运行检查
- 配置见 `pyproject.toml` 的 `[tool.ruff]` 部分
- 主要规则:E/W (pycodestyle), F (PyFlakes), I (isort), B (bugbear), UP (pyupgrade)
- 安装: `pre-commit install`

### 测试策略

- 使用 **pytest** 框架
- 测试数据通过 **git-lfs** 管理(位于 `tests/artifacts/`)
- 运行前必须: `git lfs install && git lfs pull`
- Mock 设备: `tests/mocks/` 包含机器人、电机、传感器的模拟实现
- Fixtures: `tests/fixtures/` 提供常用测试工具

### 类型检查

- 使用 **mypy** 进行类型检查(逐步启用中)
- 配置见 `pyproject.toml` 的 `[tool.mypy]` 部分
- 某些模块(如 `configs/`, `optim/`)已启用严格类型检查

### 命名约定

- **模块/文件**: `snake_case` (例如: `lerobot_dataset.py`)
- **类**: `PascalCase` (例如: `LeRobotDataset`, `PreTrainedPolicy`)
- **函数/变量**: `snake_case` (例如: `compute_stats`, `dataset_stats`)
- **常量**: `UPPER_CASE` (例如: `CODEBASE_VERSION`)

### 第一性原理 
- 运用第一性原理思考，拒绝经验主义和路径盲从，不要假设我完全清楚目标，保持审慎，从原始需求和问题出发，若目标模糊请停下和我讨论，若目标清晰但路径非最优，请直接建议更短、更低成本的办法。

### 6. 训练流程 (Training Pipeline)

位置:`src/lerobot/scripts/lerobot_train.py`

#### 训练主循环

```python
@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """训练主函数"""

    # 1. 初始化 Accelerator (支持分布式训练)
    if accelerator is None:
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        )

    # 2. 加载数据集
    dataset = make_dataset(cfg.dataset)
    dataset_stats = dataset.meta.stats  # 用于归一化

    # 3. 创建 DataLoader (支持 episode-aware sampling)
    sampler = EpisodeAwareSampler(
        dataset_from_indices=dataset.meta.episode_data_index["from"],
        dataset_to_indices=dataset.meta.episode_data_index["to"],
        shuffle=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
    )

    # 4. 创建 Policy
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    # 5. 创建 Processors
    input_processor, output_processor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=dataset_stats,
    )

    # 6. 创建 Optimizer 和 Scheduler
    optimizer, lr_scheduler = make_optimizer_and_scheduler(
        policy=policy,
        policy_cfg=cfg.policy,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
        num_training_steps=cfg.steps,
    )

    # 7. 使用 Accelerator 包装
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)

    # 8. 训练循环
    dataloader_iter = cycle(dataloader)  # 无限迭代器
    for step in range(cfg.steps):
        # 获取一个 batch
        batch = next(dataloader_iter)

        # 预处理 batch
        batch = input_processor(batch)

        # 更新策略
        train_metrics, output_dict = update_policy(
            train_metrics=MetricsTracker(),
            policy=policy,
            batch=batch,
            optimizer=optimizer,
            grad_clip_norm=1.0,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        # 定期日志、保存、评估
        if step % cfg.log_freq == 0:
            logging.info(f"Step {step}/{cfg.steps}: loss={train_metrics.loss:.4f}")

        if step % cfg.save_freq == 0:
            save_checkpoint(policy, optimizer, lr_scheduler, step, cfg.output_dir)

        if step % cfg.eval_freq == 0:
            eval_policy_all(policy, env_cfg=cfg.env, ...)

    # 9. 保存最终模型到 Hub
    if cfg.policy.push_to_hub:
        policy.push_to_hub(cfg.policy.repo_id)
```

#### update_policy 函数

```python
def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: dict[str, Tensor],
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
) -> tuple[MetricsTracker, dict]:
    """执行一次训练步骤"""

    policy.train()

    # 使用 Accelerator 的混合精度上下文
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)

    # 反向传播
    accelerator.backward(loss)

    # 梯度裁剪
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)

    # 优化器步进
    optimizer.step()
    optimizer.zero_grad()

    # 学习率调度
    if lr_scheduler is not None:
        lr_scheduler.step()

    # 记录指标
    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]

    return train_metrics, output_dict
```

**关键特性:**
- **Accelerator**:自动处理分布式训练、混合精度、梯度累积
- **EpisodeAwareSampler**:支持按 episode 边界采样,避免跨 episode 采样
- **Checkpoint 管理**:支持恢复训练 (`--resume`)
- **RA-BC (Reward-Aligned BC)**:支持基于奖励的样本加权

### 7. Robot 抽象层

位置:`src/lerobot/robots/robot.py`

#### Robot 基类

```python
class Robot(abc.ABC):
    """所有机器人的抽象基类"""

    # 必须在子类中定义
    config_class: type[RobotConfig]
    name: str

    def __init__(self, config: RobotConfig):
        self.robot_type = self.name
        self.id = config.id
        self.calibration_dir = HF_LEROBOT_CALIBRATION / ROBOTS / self.name
        self.calibration: dict[str, MotorCalibration] = {}

    # 必须实现的抽象方法
    @property
    @abc.abstractmethod
    def observation_features(self) -> dict:
        """
        定义机器人的观测特征结构

        返回示例:
        {
            "observation.state": (10,),  # 10维关节状态
            "observation.images.top": (3, 480, 640),  # RGB 相机
            "observation.images.wrist": (3, 240, 320),
        }
        """
        pass

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        """
        定义机器人的动作特征结构

        返回示例:
        {
            "action": (7,),  # 7维动作 (6 DoF + gripper)
        }
        """
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """机器人是否已连接"""
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """建立与机器人的连接"""
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """机器人是否已校准"""
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """校准机器人 (例如:记录电机零点)"""
        pass

    @abc.abstractmethod
    def configure(self) -> None:
        """配置机器人参数 (例如:设置 PID 增益)"""
        pass

    @abc.abstractmethod
    def get_observation(self) -> RobotObservation:
        """
        获取当前观测

        返回:
            RobotObservation: 字典,键与 observation_features 匹配
        """
        pass

    @abc.abstractmethod
    def send_action(self, action: RobotAction) -> None:
        """
        发送动作指令

        Args:
            action: 字典,键与 action_features 匹配
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开与机器人的连接"""
        pass
```

**实现示例** (Koch 机器人):

```python
class KochRobot(Robot):
    config_class = KochRobotConfig
    name = "koch"

    def __init__(self, config: KochRobotConfig):
        super().__init__(config)
        self.leader_arms = {}  # Dynamixel motor buses
        self.follower_arms = {}
        self.cameras = {}

    @property
    def observation_features(self) -> dict:
        return {
            "observation.state": (6,),  # 6 DoF arm
            "observation.images.top": (3, 480, 640),
        }

    @property
    def action_features(self) -> dict:
        return {"action": (6,)}

    def connect(self, calibrate: bool = True):
        # 连接电机和相机
        self.leader_arms["main"] = MotorBus(port="/dev/ttyUSB0", motors=[...])
        self.cameras["top"] = OpenCVCamera(index=0)
        if calibrate and not self.is_calibrated:
            self.calibrate()

    def get_observation(self) -> RobotObservation:
        # 读取关节位置和图像
        state = self.follower_arms["main"].read("present_position")
        image = self.cameras["top"].read()
        return {
            "observation.state": torch.tensor(state),
            "observation.images.top": torch.from_numpy(image),
        }

    def send_action(self, action: RobotAction):
        # 发送目标位置
        self.follower_arms["main"].write("goal_position", action["action"])
```

### 8. 数据集架构 (LeRobotDataset v3.0)

位置:`src/lerobot/datasets/lerobot_dataset.py`

#### 数据格式

```
dataset_repo/
├── meta/
│   ├── info.json              # 数据集元信息 (fps, codebase_version等)
│   ├── episodes.jsonl         # Episode 索引 (起止帧、长度、任务)
│   └── tasks.jsonl            # 任务定义
├── data/
│   ├── chunk-000/
│   │   ├── episode_0.parquet  # 表格数据 (state, action, reward等)
│   │   ├── observation.images.top/
│   │   │   ├── episode_0.mp4  # 视频 (或独立图像)
│   │   └── ...
│   └── ...
└── stats.safetensors          # 数据集统计 (mean, std, min, max)
```

#### LeRobotDataset 使用

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 1. 从 Hub 加载
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# 2. 访问数据
sample = dataset[0]  # 字典,包含:
# {
#     "observation.state": (n_obs_steps, state_dim),
#     "observation.images.top": (n_obs_steps, C, H, W),
#     "action": (action_horizon, action_dim),
#     "episode_index": int,
#     "frame_index": int,
#     "timestamp": float,
# }

# 3. 获取统计信息
stats = dataset.meta.stats  # {"observation.state": {"mean": ..., "std": ...}, ...}

# 4. Episode 信息
episodes = dataset.meta.episode_data_index
print(f"Total episodes: {len(episodes)}")
print(f"Episode 0: frames {episodes['from'][0]} to {episodes['to'][0]}")

# 5. 计算统计 (如果不存在)
dataset.compute_stats()

# 6. 数据集操作
# 删除 episodes: dataset.delete_episodes([0, 1, 2])
# 分割数据集: train_ds, val_ds = dataset.split(train_ratio=0.9)
# 合并数据集: merged_ds = LeRobotDataset.merge([ds1, ds2])
```

**关键特性:**
- **高效存储**:视频压缩 (H264/HEVC),Parquet 列式存储
- **流式加载**:支持从 Hub 流式加载大型数据集
- **自动解码**:视频帧自动解码为 numpy/torch 数组
- **统计计算**:自动计算 mean/std/min/max (用于归一化)

## 常见任务

### 添加新策略 (Policy)

参考 `docs/how_to_add_new_policy.md` 获取详细指南。

**快速检查清单:**
1. ✅ 创建三个文件:`configuration_*.py`, `modeling_*.py`, `processor_*.py`
2. ✅ Configuration 使用 `@PreTrainedConfig.register_subclass("name")` 装饰器
3. ✅ Policy 类定义 `config_class` 和 `name` 属性
4. ✅ 实现所有抽象方法:`forward()`, `select_action()`, `reset()`, `get_optim_params()`
5. ✅ 注册到工厂:`policies/factory.py` (三个函数)
6. ✅ 在 `policies/__init__.py` 导出配置类
7. ✅ 添加测试:`tests/policies/test_my_policy.py`

**常见错误:**
- ❌ 忘记使用 `@register_subclass` 装饰器
- ❌ 处理器步骤顺序错误 (归一化应在 tokenizer 之前)
- ❌ `input_features` 和 `output_features` 未正确设置
- ❌ `normalization_mapping` 未定义或不匹配

### 添加新机器人硬件

1. 创建 `src/lerobot/robots/my_robot/` 目录
2. 继承 `Robot` 基类,实现所有抽象方法
3. 定义 `observation_features` 和 `action_features`
4. 实现 `connect()`, `get_observation()`, `send_action()`
5. 创建配置类 (继承 `RobotConfig`)
6. 在 `robots/__init__.py` 中注册

### 添加新仿真环境

1. 实现 Gymnasium 兼容接口 (可选:继承 `gym.Env`)
2. 定义 `observation_space` 和 `action_space`
3. 在 `envs/` 中创建环境包装器
4. 注册到 `envs/factory.py`
5. 可选:上传到 HF Hub 作为 EnvHub 环境

### 调试技巧

- **可视化数据**: `lerobot-dataset-viz --dataset.repo_id=...`
- **检查设备连接**: `lerobot-find-cameras`, `lerobot-find-port`
- **调试训练**: 使用小 `batch_size=2` 和 `steps=100` 快速迭代
- **Profile 性能**: 启用 `torch.compile` 或使用 `torch.profiler`
- **检查梯度**: 在 `forward()` 后检查 `loss.requires_grad`
- **验证处理器**: 手动运行 `preprocessor(sample)` 检查输出形状

## 依赖管理

### 核心依赖

- **PyTorch** (>=2.2.1): 深度学习框架
- **Hugging Face 生态**: `datasets`, `diffusers`, `transformers`, `accelerate`
- **视觉**: `opencv-python-headless`, `av`, `torchcodec` (视频处理)
- **机器人**: `gymnasium`, `rerun-sdk` (可视化)

### 可选依赖 (extras)

策略相关:
- `[pi]`: Pi0/Pi0.5 模型 (需要自定义 transformers 分支)
- `[groot]`: GR00T N1.5 模型 (需要 flash-attn)
- `[smolvla]`, `[xvla]`: VLA 模型
- `[hilserl]`: HIL-SERL 强化学习

硬件相关:
- `[dynamixel]`, `[feetech]`: 电机支持
- `[intelrealsense]`: RealSense 相机
- `[gamepad]`, `[phone]`: 远程操作设备

仿真环境:
- `[aloha]`, `[libero]`, `[metaworld]`, `[pusht]`: 各种仿真环境

安装示例:
```bash
# 基础安装
pip install lerobot

# 开发安装(推荐)
pip install -e ".[dev,test]"

# 安装特定功能
pip install -e ".[pi,libero,dynamixel]"
```

## 资源链接

- **文档**: https://huggingface.co/docs/lerobot/index
- **Hub**: https://huggingface.co/lerobot (数据集和预训练模型)
- **Discord**: https://discord.gg/q8Dzzpym3f
- **Contributing**: 参考 `CONTRIBUTING.md`
