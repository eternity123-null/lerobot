#!/usr/bin/env python
"""
测试 CARP 基础集成到 LeRobot 框架
验证：
1. 配置类注册和创建
2. 策略类导入
3. 处理器创建
"""

import torch
import sys
from pathlib import Path

# Add lerobot to path
lerobot_path = Path(__file__).parent.parent
sys.path.insert(0, str(lerobot_path / "src"))

def test_config_registration():
    """测试配置类是否正确注册"""
    print("=" * 60)
    print("测试 1: 配置类注册")
    print("=" * 60)

    try:
        from lerobot.policies import CARPConfig
        from lerobot.configs.policies import PreTrainedConfig

        # 测试通过 ChoiceRegistry 获取
        config_cls = PreTrainedConfig.get_choice_class("carp")
        assert config_cls == CARPConfig, "配置类注册失败"
        print("✓ 配置类注册成功")

        # 测试创建配置
        config = CARPConfig()
        print(f"✓ 配置创建成功: training_stage={config.training_stage}")

        # 验证必需的方法
        assert hasattr(config, "get_optimizer_preset"), "缺少 get_optimizer_preset"
        assert hasattr(config, "get_scheduler_preset"), "缺少 get_scheduler_preset"
        assert hasattr(config, "validate_features"), "缺少 validate_features"
        print("✓ 必需方法存在")

        return True
    except Exception as e:
        print(f"✗ 配置类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_import():
    """测试策略类导入"""
    print("\n" + "=" * 60)
    print("测试 2: 策略类导入")
    print("=" * 60)

    try:
        from lerobot.policies.factory import get_policy_class

        # 测试 VAE 策略
        vae_cls = get_policy_class("carp_vae")
        print(f"✓ VAE 策略类导入成功: {vae_cls.__name__}")

        # 测试 AR 策略
        ar_cls = get_policy_class("carp")
        print(f"✓ AR 策略类导入成功: {ar_cls.__name__}")

        return True
    except Exception as e:
        print(f"✗ 策略类导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_processor_creation():
    """测试处理器创建"""
    print("\n" + "=" * 60)
    print("测试 3: 处理器创建")
    print("=" * 60)

    try:
        from lerobot.policies import CARPConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.configs.types import PolicyFeature, FeatureType

        # 创建测试配置
        config = CARPConfig(
            n_obs_steps=1,
            action_horizon=16,
            device="cpu",
        )

        # 设置特征
        config.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }
        config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }

        # 创建处理器（不使用统计信息）
        preprocessor, postprocessor = make_pre_post_processors(config)
        print(f"✓ 预处理器创建成功: {len(preprocessor.steps)} 个步骤")
        print(f"✓ 后处理器创建成功: {len(postprocessor.steps)} 个步骤")

        return True
    except Exception as e:
        print(f"✗ 处理器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae_model_creation():
    """测试 VAE 模型创建"""
    print("\n" + "=" * 60)
    print("测试 4: VAE 模型创建")
    print("=" * 60)

    try:
        from lerobot.policies import CARPConfig
        from lerobot.policies.carp.modeling_carp_vae import CARPVAEPolicy
        from lerobot.configs.types import PolicyFeature, FeatureType

        # 创建配置
        config = CARPConfig(
            training_stage="vae",
            n_obs_steps=1,
            action_horizon=16,
            vocab_size=512,
            vocab_ch=8,
            device="cpu",
        )

        # 设置特征
        config.input_features = {}
        config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }

        # 创建模型
        policy = CARPVAEPolicy(config)
        print(f"✓ VAE 模型创建成功")
        print(f"  - 动作维度: {policy.action_dim}")
        print(f"  - VAE 参数量: {sum(p.numel() for p in policy.vae.parameters()):,}")

        # 测试前向传播
        batch = {
            "action": torch.randn(2, 16, 7)  # (B, T, A)
        }
        loss, loss_dict = policy.forward(batch)
        print(f"✓ VAE 前向传播成功")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Loss dict: {loss_dict}")

        return True
    except Exception as e:
        print(f"✗ VAE 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("CARP 基础集成测试")
    print("=" * 60)

    results = []

    # 运行所有测试
    results.append(("配置注册", test_config_registration()))
    results.append(("策略导入", test_policy_import()))
    results.append(("处理器创建", test_processor_creation()))
    results.append(("VAE模型创建", test_vae_model_creation()))

    # 总结
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
        print("\n🎉 所有测试通过！CARP 基础集成成功！")
        return 0
    else:
        print(f"\n❌ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
