#!/usr/bin/env python
"""
CARP Stage 2: Autoregressive Transformer Training Script

This script trains the Coarse-to-Fine Autoregressive model to predict
multi-scale action tokens from observations, using a frozen VAE decoder.

Requirements:
- Pretrained VAE checkpoint from Stage 1
- Dataset with observations (images + state) and actions

Usage:
    python train_carp_ar.py \\
        --dataset_path=/path/to/dataset \\
        --vae_checkpoint=outputs/carp_vae/vae_final.pt \\
        --output_dir=outputs/carp_ar
"""

import logging
import sys
from pathlib import Path

# Add lerobot to path
lerobot_path = Path(__file__).parent
sys.path.insert(0, str(lerobot_path / "src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies import CARPConfig
from lerobot.policies.carp.modeling_carp import CARPPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataloader(dataset_path: str, batch_size: int, num_workers: int = 4):
    """Create dataloader for AR training"""
    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = LeRobotDataset(dataset_path)
    logger.info(f"Dataset loaded: {len(dataset)} frames, {len(dataset.meta.episodes)} episodes")

    # Create episode-aware sampler
    sampler = EpisodeAwareSampler(
        dataset_from_indices=dataset.meta.episodes["dataset_from_index"],
        dataset_to_indices=dataset.meta.episodes["dataset_to_index"],
        shuffle=True,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=True,
    )

    return dataset, dataloader


def train_ar(
    dataset_path: str,
    vae_checkpoint_path: str,
    output_dir: str,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    save_freq: int = 10,
    log_freq: int = 100,
):
    """Train CARP Autoregressive Transformer"""

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset, dataloader = create_dataloader(dataset_path, batch_size)

    # Infer features
    features = dataset_to_policy_features(dataset.meta.features)

    # Create config
    config = CARPConfig(
        training_stage="ar",
        action_horizon=16,
        n_obs_steps=1,
        vocab_size=512,
        vocab_ch=8,
        ar_depth=32,
        ar_embed_dim=160,
        ar_num_heads=32,
        ar_lr=learning_rate,
        vae_checkpoint_path=vae_checkpoint_path,
        device=device,
        task_num=10,  # LIBERO has ~10 tasks
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

    # Create policy
    logger.info("Creating AR policy...")
    logger.info("Note: This will load pretrained VAE and freeze it")
    policy = CARPPolicy(config)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen VAE parameters: {total_params - trainable_params:,}")

    # Create optimizer (only for trainable params)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=learning_rate,
        weight_decay=config.ar_weight_decay,
        betas=[0.9, 0.999],
    )

    # Create processors
    dataset_stats = dataset.meta.stats if hasattr(dataset.meta, 'stats') else None
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        dataset_stats=dataset_stats,
    )

    # Training loop
    logger.info("Starting AR training...")
    global_step = 0

    for epoch in range(num_epochs):
        policy.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Preprocess batch
            batch = preprocessor(batch)

            # Extract task IDs if available
            if "task_index" in batch:
                batch["task_id"] = batch["task_index"]

            # Forward pass
            # Note: In real training, we need to:
            # 1. Sample action sequences (action_horizon steps)
            # 2. Handle n_obs_steps observation history
            # For this simplified version, we skip these complexities

            # TODO: Implement proper action sequence sampling
            # TODO: Implement observation history handling

            try:
                loss, loss_dict = policy.forward(batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                # Logging
                epoch_loss += loss.item()
                epoch_accuracy += loss_dict.get("accuracy", 0.0)

                if global_step % log_freq == 0:
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{loss_dict.get('accuracy', 0.0):.4f}",
                    })

                global_step += 1

            except Exception as e:
                logger.warning(f"Skipping batch due to error: {e}")
                continue

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        avg_accuracy = epoch_accuracy / len(dataloader)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}"
        )

        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "ar_model_state_dict": policy.ar_model.state_dict(),
                "obs_encoder_state_dict": policy.obs_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = output_dir / "ar_final.pt"
    torch.save({
        "ar_model_state_dict": policy.ar_model.state_dict(),
        "obs_encoder_state_dict": policy.obs_encoder.state_dict(),
        "config": config,
    }, final_path)
    logger.info(f"Training complete! Final model saved: {final_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Train CARP AR Transformer")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--vae_checkpoint", type=str, required=True,
                        help="Path to pretrained VAE checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/carp_ar",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--save_freq", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log_freq", type=int, default=100,
                        help="Log every N steps")

    args = parser.parse_args()

    train_ar(
        dataset_path=args.dataset_path,
        vae_checkpoint_path=args.vae_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
    )


if __name__ == "__main__":
    main()
