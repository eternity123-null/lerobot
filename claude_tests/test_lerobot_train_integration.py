#!/usr/bin/env python
"""
测试 CARP 与 lerobot-train 的集成

验证：
1. 配置加载正确
2. 数据处理器工作正常
3. Policy forward() 能够处理正确的数据维度
4. 模拟完整的训练步骤
"""

import sys
from pathlib import Path

# Add lerobot to path
lerobot_path = Path(__file__).parent
sys.path.insert(0, str(lerobot_path / "src"))

import torch
from lerobot.policies import CARPConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType


def test_carp_vae_integration():
    """测试 CARP VAE 与 lerobot-train 的集成"""
    print("=" * 60)
    print("测试 1: CARP VAE 集成")
    print("=" * 60)

    try:
        # 创建配置（模拟 lerobot-train 的配置创建）
        config = CARPConfig(
            training_stage="vae",
            action_horizon=16,
            n_obs_steps=1,
            vocab_size=512,
            vocab_ch=8,
            device="cpu",
        )

        # 模拟特征推断（来自数据集）
        from lerobot.configs.types import PolicyFeature, FeatureType

        config.input_features = {}  # VAE 不需要观测
        config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))
        }

        # 创建 Policy（使用工厂方法）
        from lerobot.policies.factory import get_policy_class
        policy_cls = get_policy_class("carp_vae")
        policy = policy_cls(config)

        print(f"✓ VAE Policy 创建成功: {policy_cls.__name__}")
        print(f"  - 参数量: {sum(p.numel() for p in policy.parameters()):,}")

        # 创建处理器
        preprocessor, postprocessor = make_pre_post_processors(config)
        print(f"✓ 处理器创建成功")
        print(f"  - 预处理步骤: {len(preprocessor.steps)}")
        print(f"  - 后处理步骤: {len(postprocessor.steps)}")

        # 模拟训练批次
        batch = {
            "action": torch.randn(4, 7),  # (B, action_dim) - 单帧动作
        }

        print(f"\n原始批次:")
        print(f"  - action: {batch['action'].shape}")

        # 预处理
        batch = preprocessor(batch)
        print(f"\n预处理后:")
        print(f"  - action: {batch['action'].shape}")

        # 检查维度
        expected_shape = (4, config.action_horizon, 7)  # (B, T, A)
        actual_shape = batch['action'].shape

        if actual_shape == expected_shape:
            print(f"✓ 动作维度正确: {actual_shape}")
        else:
            print(f"✗ 动作维度错误: 期望 {expected_shape}, 实际 {actual_shape}")
            return False

        # Forward pass
        loss, loss_dict = policy.forward(batch)
        print(f"\n✓ Forward pass 成功")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Loss dict: {list(loss_dict.keys())}")

        return True

    except Exception as e:
        print(f"\n✗ VAE 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_carp_ar_integration():
    """测试 CARP AR 与 lerobot-train 的集成"""
    print("\n" + "=" * 60)
    print("测试 2: CARP AR 集成")
    print("=" * 60)

    try:
        # 创建临时 VAE checkpoint
        import tempfile
        vae_checkpoint = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        vae_checkpoint_path = vae_checkpoint.name
        vae_checkpoint.close()

        # 保存一个假的 VAE checkpoint
        from lerobot.policies.carp.MSAT.vqvae import MultiScaleActionTokenizer

        vae = MultiScaleActionTokenizer(
            vocab_size=512,
            z_channels=8,
            ch=2,
            action_dim=7,
            num_actions=16,
            dropout=0.0,
            beta=0.25,
            using_znorm=True,
            quant_conv_ks=3,
            quant_resi=0.5,
            share_quant_resi=4,
            v_patch_nums=(1, 2, 3, 4),
            test_mode=True,
        )

        torch.save({
            "model_state_dict": vae.state_dict(),
        }, vae_checkpoint_path)

        print(f"✓ 临时 VAE checkpoint 创建: {vae_checkpoint_path}")

        # 创建配置
        config = CARPConfig(
            training_stage="ar",
            action_horizon=16,
            n_obs_steps=1,
            vocab_size=512,
            vocab_ch=8,
            ar_depth=4,  # 使用较小的深度进行测试
            ar_embed_dim=160,
            ar_num_heads=4,  # 使用较小的头数
            vae_checkpoint_path=vae_checkpoint_path,
            device="cpu",
            task_num=10,
        )

        # 设置特征
        from lerobot.configs.types import PolicyFeature, FeatureType

        config.input_features = {
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        }
        config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))
        }

        # 创建 Policy
        from lerobot.policies.factory import get_policy_class
        policy_cls = get_policy_class("carp")

        print(f"创建 AR Policy（需要加载 VAE）...")
        policy = policy_cls(config)

        print(f"✓ AR Policy 创建成功: {policy_cls.__name__}")
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        print(f"  - 冻结参数: {total_params - trainable_params:,}")

        # 创建处理器
        preprocessor, postprocessor = make_pre_post_processors(config)
        print(f"✓ 处理器创建成功")

        # 模拟训练批次
        batch = {
            "observation.images.image": torch.randn(2, 3, 256, 256),
            "observation.state": torch.randn(2, 8),
            "action": torch.randn(2, 7),
            "task_index": torch.tensor([0, 1]),
        }

        print(f"\n原始批次:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.shape}")

        # 预处理
        batch = preprocessor(batch)
        print(f"\n预处理后:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.shape}")

        # Forward pass
        print(f"\n执行 forward pass...")
        try:
            loss, loss_dict = policy.forward(batch)
            print(f"✓ Forward pass 成功")
            print(f"  - Loss: {loss.item():.4f}")
            print(f"  - Loss dict: {list(loss_dict.keys())}")
        except Exception as e:
            print(f"✗ Forward pass 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Cleanup
        Path(vae_checkpoint_path).unlink()

        return True

    except Exception as e:
        print(f"\n✗ AR 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("CARP + lerobot-train 集成测试")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("VAE 集成", test_carp_vae_integration()))
    results.append(("AR 集成", test_carp_ar_integration()))

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
        print("\n🎉 所有集成测试通过！可以使用 lerobot-train！")
        return 0
    else:
        print(f"\n❌ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
