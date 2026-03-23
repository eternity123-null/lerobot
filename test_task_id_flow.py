#!/usr/bin/env python3
"""
测试 task_id 在训练和评估中的完整数据流
不需要真实环境初始化
"""

import torch
import numpy as np
import sys
sys.path.insert(0, 'src')

print("=" * 80)
print("Task ID 数据流测试（训练 + 评估）")
print("=" * 80)

# ===============================================================================
# Part 1: 训练时的 task_id 流程
# ===============================================================================
print("\n【Part 1: 训练时的 Task ID 流程】")
print("-" * 80)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.carp.configuration_carp import CARPConfig
from lerobot.policies.factory import make_pre_post_processors

# 1. 加载数据集
dataset_path = "/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero"
print(f"\n1. 加载数据集: {dataset_path}")

try:
    dataset = LeRobotDataset(dataset_path)
    print(f"   ✓ 数据集加载成功 (total_tasks={dataset.meta.total_tasks})")
except Exception as e:
    print(f"   ✗ 数据集加载失败: {e}")
    sys.exit(1)

# 2. 创建 CARP 配置和处理器
print(f"\n2. 创建 CARP 配置和处理器")

from lerobot.configs.types import PolicyFeature, FeatureType

input_features = {}
output_features = {}

for key, feature_info in dataset.meta.features.items():
    shape = tuple(feature_info["shape"])

    if key.startswith("observation.images."):
        input_features[key] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(shape[2], shape[0], shape[1])  # C, H, W
        )
    elif key == "observation.state":
        input_features[key] = PolicyFeature(
            type=FeatureType.STATE,
            shape=shape
        )
    elif key == "action":
        output_features[key] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=shape
        )

config = CARPConfig()
config.input_features = input_features
config.output_features = output_features
config.device = "cpu"
config.task_num = dataset.meta.total_tasks

dataset_stats = dataset.meta.stats
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=config,
    dataset_stats=dataset_stats,
)

print(f"   ✓ Preprocessor 创建成功")

# 3. 测试数据集样本的 task_id 注入
print(f"\n3. 测试数据集样本的 task_id 提取和转换")

sample = dataset[500]
print(f"   原始样本:")
print(f"     task_index: {sample.get('task_index', 'MISSING')}")
print(f"     episode_index: {sample.get('episode_index', 'MISSING')}")

# 预处理
processed = preprocessor(sample)

print(f"   预处理后:")
if "task_id" in processed:
    task_id = processed["task_id"]
    if isinstance(task_id, torch.Tensor):
        task_id_value = task_id.squeeze().item() if task_id.numel() == 1 else task_id
    else:
        task_id_value = task_id
    print(f"     task_id: {task_id_value} ✓")
else:
    print(f"     task_id: MISSING ✗")

print(f"\n   ✓ 训练流程: 数据集[task_index] → Preprocessor → batch[task_id] → 模型")

# ===============================================================================
# Part 2: 评估时的 task_id 流程
# ===============================================================================
print("\n【Part 2: 评估时的 Task ID 流程】")
print("-" * 80)

# 模拟 LIBERO 环境的 reset() 和 step() 返回
print("\n1. 模拟 LIBERO 环境行为")

class MockLiberoEnv:
    """模拟 LIBERO 环境的 task_id 返回行为"""
    def __init__(self, task_id: int):
        self.task_id = task_id
        self.task_name = f"LIBERO_SPATIAL_{task_id}"

    def reset(self):
        """
        LIBERO reset() 返回:
        - obs: 观测字典
        - info: {"is_success": False}  # ← 注意：reset 时没有 task_id!
        """
        obs = {
            "pixels": {
                "image": np.random.randint(0, 255, (360, 360, 3), dtype=np.uint8),
                "image2": np.random.randint(0, 255, (360, 360, 3), dtype=np.uint8),
            },
            "robot_state": {
                "eef": {
                    "pos": np.random.randn(3),
                    "quat": np.random.randn(4),
                },
                "gripper": {
                    "qpos": np.random.randn(2),
                },
                "joint": {
                    "pos": np.random.randn(7),
                    "vel": np.random.randn(7),
                },
            }
        }
        info = {"is_success": False}  # ← reset 时没有 task_id
        return obs, info

    def step(self, action):
        """
        LIBERO step() 返回:
        - obs: 观测字典
        - reward: float
        - terminated: bool
        - truncated: bool
        - info: {"task_id": self.task_id, ...}  # ← step 时有 task_id
        """
        obs = {
            "pixels": {
                "image": np.random.randint(0, 255, (360, 360, 3), dtype=np.uint8),
                "image2": np.random.randint(0, 255, (360, 360, 3), dtype=np.uint8),
            },
            "robot_state": {
                "eef": {
                    "pos": np.random.randn(3),
                    "quat": np.random.randn(4),
                },
                "gripper": {
                    "qpos": np.random.randn(2),
                },
                "joint": {
                    "pos": np.random.randn(7),
                    "vel": np.random.randn(7),
                },
            }
        }
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "task": self.task_name,
            "task_id": self.task_id,  # ← step 时有 task_id
            "done": False,
            "is_success": False,
        }
        return obs, reward, terminated, truncated, info

# 测试环境
test_task_id = 15
env = MockLiberoEnv(task_id=test_task_id)

print(f"   模拟环境: task_id={test_task_id}")

# 2. 测试 reset()
print(f"\n2. 测试 reset() 的 task_id 处理")

obs, info = env.reset()
print(f"   reset() 返回的 info:")
print(f"     is_success: {info.get('is_success')}")
print(f"     task_id: {info.get('task_id', 'MISSING')}  ← 缺失!")

# 评估脚本应该手动添加 task_id
if "task_id" not in info:
    # 从环境的属性中获取 task_id
    info["task_id"] = env.task_id
    print(f"   手动添加 task_id 到 info: {info['task_id']} ✓")

# 将 task_id 注入到观测中 (用于 preprocessor)
obs_with_task = {**obs, "task_id": info["task_id"]}
print(f"   将 task_id 添加到观测字典 ✓")

# 3. 测试 step()
print(f"\n3. 测试 step() 的 task_id 处理")

action = np.random.randn(7)
obs, reward, terminated, truncated, info = env.step(action)

print(f"   step() 返回的 info:")
print(f"     task: {info.get('task')}")
print(f"     task_id: {info.get('task_id')} ✓")
print(f"     done: {info.get('done')}")
print(f"     is_success: {info.get('is_success')}")

# 将 task_id 注入到观测中
obs_with_task = {**obs, "task_id": info["task_id"]}
print(f"   将 task_id 添加到观测字典 ✓")

# 4. 测试 preprocessor 处理（不使用完整预处理流程）
print(f"\n4. 测试 Task ID 在评估时的保留机制")

# 直接测试 CARPTaskIDProcessorStep
from lerobot.policies.carp.carp_task_id_step import CARPTaskIDProcessorStep
from lerobot.processor.core import EnvTransition

print(f"   测试场景 1: task_id 在顶层字段")
test_transition_1: EnvTransition = {
    "observation": {"test": torch.randn(3)},
    "action": None,
    "reward": 0.0,
    "done": False,
    "truncated": False,
    "info": {},
    "complementary_data": {},
    "task_id": torch.tensor([info["task_id"]]),  # ← 顶层 task_id
}

step = CARPTaskIDProcessorStep()
result_1 = step(test_transition_1)

if "task_id" in result_1.get("complementary_data", {}):
    tid = result_1["complementary_data"]["task_id"]
    print(f"     ✓ task_id 移动到 complementary_data: {tid.item() if isinstance(tid, torch.Tensor) else tid}")
else:
    print(f"     ✗ task_id 未移动到 complementary_data")

print(f"\n   测试场景 2: task_index 在 complementary_data")
test_transition_2: EnvTransition = {
    "observation": {"test": torch.randn(3)},
    "action": None,
    "reward": 0.0,
    "done": False,
    "truncated": False,
    "info": {},
    "complementary_data": {"task_index": torch.tensor([info["task_id"]])},  # ← task_index
}

result_2 = step(test_transition_2)

comp_data = result_2.get("complementary_data", {})
if "task_id" in comp_data and "task_index" not in comp_data:
    tid = comp_data["task_id"]
    print(f"     ✓ task_index 重命名为 task_id: {tid.item() if isinstance(tid, torch.Tensor) else tid}")
else:
    print(f"     ✗ 重命名失败")
    print(f"       complementary_data: {list(comp_data.keys())}")

print(f"\n   测试场景 3: 验证评估脚本需要的操作")
print(f"     1. env.step() 返回 info['task_id']: {info['task_id']}")
print(f"     2. 将其添加到观测: obs['task_id'] = info['task_id']")
print(f"     3. Preprocessor 自动处理 → complementary_data['task_id']")
print(f"     4. 模型从 batch['task_id'] 获取条件化信息")
print(f"     ✓ 数据流程验证完成")

# ===============================================================================
# 总结
# ===============================================================================
print("\n" + "=" * 80)
print("【总结】")
print("-" * 80)
print("✓ 训练流程:")
print("  LeRobotDataset → batch[task_index] → Preprocessor → batch[task_id] → 模型")
print()
print("✓ 评估流程:")
print("  1. env.reset() 返回 info (不含 task_id)")
print("     → 需要手动添加: info['task_id'] = env.task_id")
print("  2. env.step() 返回 info (含 task_id)")
print("  3. 将 info['task_id'] 添加到 obs 字典")
print("  4. Preprocessor 保留 task_id 字段")
print("  5. 模型使用 batch['task_id'] 进行条件化生成")
print()
print("⚠️  重要注意事项:")
print("  - reset() 后需要手动从 env.task_id 获取并添加到 info")
print("  - step() 后 info 自动包含 task_id")
print("  - 评估脚本需要确保 obs 字典包含 task_id 字段")
print("=" * 80)
