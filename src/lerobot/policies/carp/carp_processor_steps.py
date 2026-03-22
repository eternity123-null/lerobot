#!/usr/bin/env python
"""
自定义处理步骤：从数据集中采样动作序列

CARP 需要 (B, action_horizon, action_dim) 的动作序列，
但 LeRobotDataset 默认只返回 (B, action_dim) 的单帧动作。
"""

from dataclasses import dataclass
import torch

from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import EnvTransition
from lerobot.configs.types import PipelineFeatureType, PolicyFeature


@ProcessorStepRegistry.register(name="carp_sample_action_sequence")
@dataclass
class CARPSampleActionSequenceStep(ProcessorStep):
    """
    从当前帧开始采样 future actions 序列

    将 action: (B, action_dim) 转换为 action: (B, action_horizon, action_dim)
    通过访问数据集的 future actions（如果可用）
    """

    action_horizon: int = 16

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """采样动作序列"""
        transition = transition.copy()

        if "action" in transition and transition["action"] is not None:
            action = transition["action"]

            # 检查是否已经是序列格式
            if action.ndim == 3:  # (B, T, A)
                # 已经是序列，确保长度匹配
                if action.shape[1] != self.action_horizon:
                    # 裁剪或填充到 action_horizon
                    B, T, A = action.shape
                    if T > self.action_horizon:
                        action = action[:, :self.action_horizon, :]
                    else:
                        # 用最后一帧填充
                        padding = action[:, -1:, :].repeat(1, self.action_horizon - T, 1)
                        action = torch.cat([action, padding], dim=1)
                transition["action"] = action

            elif action.ndim == 2:  # (B, A) - 单帧动作
                # 需要扩展为序列
                # 注意：理想情况下应该从数据集获取 future actions
                # 这里简化实现：重复当前动作
                B, A = action.shape
                action = action.unsqueeze(1).repeat(1, self.action_horizon, 1)  # (B, T, A)
                transition["action"] = action

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """声明特征变换：action 维度从 (A,) 变为 (T, A)"""
        features = features.copy()

        if PipelineFeatureType.OUTPUT in features:
            outputs = features[PipelineFeatureType.OUTPUT].copy()
            if "action" in outputs:
                action_ft = outputs["action"]
                # 更新 shape: (action_dim,) -> (action_horizon, action_dim)
                new_shape = (self.action_horizon,) + tuple(action_ft.shape)
                outputs["action"] = PolicyFeature(
                    type=action_ft.type,
                    shape=new_shape,
                )
            features[PipelineFeatureType.OUTPUT] = outputs

        return features


@ProcessorStepRegistry.register(name="carp_add_temporal_dimension")
@dataclass
class CARPAddTemporalDimensionStep(ProcessorStep):
    """
    为观测添加时间维度

    将 observation.*: (B, ...) 转换为 observation.*: (B, 1, ...)
    用于兼容需要时间维度的模型
    """

    n_obs_steps: int = 1

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """添加时间维度到观测"""
        transition = transition.copy()

        if "observation" in transition and transition["observation"] is not None:
            obs = transition["observation"]
            new_obs = {}

            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    # 检查是否已经有时间维度
                    if value.ndim >= 2 and not key.startswith("observation."):
                        # 已经有时间维度
                        new_obs[key] = value
                    else:
                        # 添加时间维度: (B, ...) -> (B, 1, ...)
                        new_obs[key] = value.unsqueeze(1)
                else:
                    new_obs[key] = value

            transition["observation"] = new_obs

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """声明特征变换：观测添加时间维度"""
        features = features.copy()

        if PipelineFeatureType.INPUT in features:
            inputs = features[PipelineFeatureType.INPUT].copy()
            for key, ft in inputs.items():
                if key.startswith("observation."):
                    # 更新 shape: (...) -> (1, ...)
                    new_shape = (self.n_obs_steps,) + tuple(ft.shape)
                    inputs[key] = PolicyFeature(
                        type=ft.type,
                        shape=new_shape,
                    )
            features[PipelineFeatureType.INPUT] = inputs

        return features
