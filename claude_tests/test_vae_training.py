#!/usr/bin/env python
"""
Quick test for CARP VAE training

Tests that the training loop can run without errors.
"""

import sys
from pathlib import Path

# Add lerobot to path
lerobot_path = Path(__file__).parent
sys.path.insert(0, str(lerobot_path / "src"))

import torch
from lerobot.policies import CARPConfig
from lerobot.policies.carp.modeling_carp_vae import CARPVAEPolicy
from lerobot.configs.types import PolicyFeature, FeatureType

def test_vae_training_loop():
    """Test VAE training with synthetic data"""
    print("=" * 60)
    print("测试 CARP VAE 训练循环")
    print("=" * 60)

    # Create config
    config = CARPConfig(
        training_stage="vae",
        action_horizon=16,
        vocab_size=512,
        vocab_ch=8,
        vch=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Set features
    config.input_features = {}
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))
    }

    # Create policy
    print("创建 VAE policy...")
    policy = CARPVAEPolicy(config)
    print(f"✓ VAE 参数量: {sum(p.numel() for p in policy.parameters()):,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=3e-4,
        weight_decay=0.005,
        betas=[0.5, 0.9],
    )

    # Create synthetic batch
    batch_size = 4
    batch = {
        "action": torch.randn(batch_size, config.action_horizon, 7).to(config.device)
    }

    print(f"\n测试前向传播...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Action shape: {batch['action'].shape}")

    # Training step
    policy.train()
    loss, loss_dict = policy.forward(batch)

    print(f"✓ 前向传播成功")
    print(f"  - Loss: {loss.item():.4f}")
    print(f"  - Loss dict: {loss_dict}")

    # Backward pass
    print(f"\n测试反向传播...")
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    print(f"✓ 反向传播成功")
    print(f"  - Gradient norm: {grad_norm:.4f}")

    # Test multiple steps
    print(f"\n测试多步训练...")
    losses = []
    for step in range(10):
        batch = {
            "action": torch.randn(batch_size, config.action_horizon, 7).to(config.device)
        }
        loss, _ = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    print(f"✓ 多步训练完成")
    print(f"  - 初始 loss: {losses[0]:.4f}")
    print(f"  - 最终 loss: {losses[-1]:.4f}")

    # Test save/load
    print(f"\n测试保存和加载...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        checkpoint_path = tmp.name

    torch.save({
        "model_state_dict": policy.vae.state_dict(),
        "config": config,
    }, checkpoint_path)
    print(f"✓ Checkpoint 保存: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    new_policy = CARPVAEPolicy(checkpoint["config"])
    new_policy.vae.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Checkpoint 加载成功")

    # Cleanup
    Path(checkpoint_path).unlink()

    print("\n" + "=" * 60)
    print("🎉 所有 VAE 训练测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        test_vae_training_loop()
        exit(0)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
