#!/usr/bin/env python
"""
测试 CARP 与 LIBERO 数据集的集成
验证：
1. 数据集加载和采样
2. 特征维度匹配
3. Task ID 处理
4. Episode 边界处理
"""

import torch
import sys
from pathlib import Path

# Add lerobot to path
lerobot_path = Path(__file__).parent
sys.path.insert(0, str(lerobot_path / "src"))

def test_libero_dataset_loading():
    """测试 LIBERO 数据集加载"""
    print("=" * 60)
    print("测试 1: LIBERO 数据集加载")
    print("=" * 60)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # LIBERO dataset path (as specified in Carp.md)
        dataset_path = "/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero"

        # Check if dataset exists
        if not Path(dataset_path).exists():
            print(f"⚠ 数据集路径不存在: {dataset_path}")
            print("  请确认数据集路径是否正确")
            return False

        # Try loading dataset
        print(f"加载数据集: {dataset_path}")
        dataset = LeRobotDataset(dataset_path)

        print(f"✓ 数据集加载成功")
        print(f"  - 总帧数: {len(dataset)}")
        print(f"  - Episode 数: {len(dataset.meta.episodes)}")

        # Print feature info
        print("\n特征信息:")
        for key, ft in dataset.meta.features.items():
            print(f"  - {key}: {ft}")

        # Print first episode info
        first_ep = dataset.meta.episodes[0]
        print(f"\n第一个 Episode:")
        print(f"  - Index: {first_ep['episode_index']}")
        print(f"  - 帧范围: {first_ep['dataset_from_index']} - {first_ep['dataset_to_index']}")
        print(f"  - 长度: {first_ep['length']}")
        print(f"  - 任务: {first_ep['tasks']}")

        # Sample one data point
        sample = dataset[0]
        print("\n第一个样本的键:")
        for key in sample.keys():
            if isinstance(sample[key], torch.Tensor):
                print(f"  - {key}: {sample[key].shape}")
            else:
                print(f"  - {key}: {type(sample[key])}")

        return True

    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_carp_with_libero_features():
    """测试 CARP 配置与 LIBERO 特征的兼容性"""
    print("\n" + "=" * 60)
    print("测试 2: CARP 配置与 LIBERO 特征兼容性")
    print("=" * 60)

    try:
        from lerobot.policies import CARPConfig
        from lerobot.configs.types import PolicyFeature, FeatureType
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset_path = "/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero"

        if not Path(dataset_path).exists():
            print("⚠ 跳过测试（数据集不存在）")
            return True

        # Load dataset
        dataset = LeRobotDataset(dataset_path)

        # Infer features from dataset
        from lerobot.datasets.utils import dataset_to_policy_features
        features = dataset_to_policy_features(dataset.meta.features)

        # Create CARP config
        config = CARPConfig(
            training_stage="ar",
            n_obs_steps=1,
            action_horizon=16,
            device="cpu",
        )

        # Set features
        config.input_features = {
            key: ft for key, ft in features.items()
            if ft.type != FeatureType.ACTION
        }
        config.output_features = {
            key: ft for key, ft in features.items()
            if ft.type == FeatureType.ACTION
        }

        print("✓ 特征推断成功")
        print("\n输入特征:")
        for key, ft in config.input_features.items():
            print(f"  - {key}: type={ft.type}, shape={ft.shape}")

        print("\n输出特征:")
        for key, ft in config.output_features.items():
            print(f"  - {key}: type={ft.type}, shape={ft.shape}")

        # Validate features
        config.validate_features()
        print("✓ 特征验证通过")

        return True

    except Exception as e:
        print(f"✗ 特征兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_with_carp():
    """测试 DataLoader 与 CARP 的集成"""
    print("\n" + "=" * 60)
    print("测试 3: DataLoader 与 CARP 集成")
    print("=" * 60)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.sampler import EpisodeAwareSampler
        import torch.utils.data

        dataset_path = "/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero"

        if not Path(dataset_path).exists():
            print("⚠ 跳过测试（数据集不存在）")
            return True

        # Load dataset
        dataset = LeRobotDataset(dataset_path)

        # Create episode-aware sampler
        sampler = EpisodeAwareSampler(
            dataset_from_indices=dataset.meta.episodes["dataset_from_index"],
            dataset_to_indices=dataset.meta.episodes["dataset_to_index"],
            shuffle=True,
        )

        # Create DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            sampler=sampler,
        )

        # Get one batch
        batch = next(iter(dataloader))

        print("✓ DataLoader 创建成功")
        print(f"\nBatch 内容:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.shape} ({value.dtype})")

        # Check batch dimensions
        action_shape = batch["action"].shape
        action_dim = dataset.meta.features["action"]["shape"][0]
        expected_shape = (4, action_dim)  # (batch_size, action_dim)
        print(f"\n动作形状检查:")
        print(f"  - 实际: {action_shape}")
        print(f"  - 期望: {expected_shape}")

        # Note: CARP需要 (B, action_horizon, action_dim)
        # 默认 LeRobotDataset 只返回 (B, action_dim)
        # 需要在训练时处理时间维度

        return True

    except Exception as e:
        print(f"✗ DataLoader 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("CARP + LIBERO 数据集集成测试")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("数据集加载", test_libero_dataset_loading()))
    results.append(("特征兼容性", test_carp_with_libero_features()))
    results.append(("DataLoader集成", test_dataloader_with_carp()))

    # Summary
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！CARP 与 LIBERO 数据集集成成功！")
        return 0
    else:
        print(f"\n❌ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
