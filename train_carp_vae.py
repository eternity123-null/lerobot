#!/usr/bin/env python
"""
CARP Stage 1: VQ-VAE Training Script

This script trains the Multi-Scale Action Tokenizer (MSAT) which learns to
encode action sequences into discrete multi-scale tokens.

Usage:
    python train_carp_vae.py --dataset.repo_id=<dataset_path> --output_dir=outputs/carp_vae
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
from lerobot.policies.carp.modeling_carp_vae import CARPVAEPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataloader(dataset_path: str, batch_size: int, num_workers: int = 4):
    """Create dataloader for VAE training"""
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
        drop_last=True,  # Drop incomplete batches
    )

    return dataset, dataloader


def train_vae(
    dataset_path: str,
    output_dir: str,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 3e-4,
    device: str = "cuda",
    save_freq: int = 10,
    log_freq: int = 100,
):
    """Train CARP VQ-VAE"""

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset, dataloader = create_dataloader(dataset_path, batch_size)

    # Infer features
    features = dataset_to_policy_features(dataset.meta.features)
    action_feature = next(ft for key, ft in features.items() if ft.type == FeatureType.ACTION)
    action_dim = action_feature.shape[0]

    logger.info(f"Action dimension: {action_dim}")

    # Create config
    config = CARPConfig(
        training_stage="vae",
        action_horizon=16,  # CARP default
        n_obs_steps=1,  # Not used in VAE training
        vocab_size=512,
        vocab_ch=8,
        vch=2,
        vae_lr=learning_rate,
        device=device,
    )

    # Set features (VAE only needs actions)
    config.input_features = {}
    config.output_features = {"action": action_feature}

    # Create policy
    logger.info("Creating VAE policy...")
    policy = CARPVAEPolicy(config)
    logger.info(f"VAE parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=learning_rate,
        weight_decay=config.vae_weight_decay,
        betas=[0.5, 0.9],
    )

    # Training loop
    logger.info("Starting VAE training...")
    global_step = 0

    for epoch in range(num_epochs):
        policy.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Extract actions and expand to action_horizon
            # Note: Default dataset returns (B, action_dim), need (B, action_horizon, action_dim)
            actions = batch["action"]  # (B, action_dim)

            # For VAE training, we need to create action sequences
            # Option 1: Use future actions if available
            # Option 2: Repeat current action (for testing)
            # TODO: Implement proper action sequence extraction from dataset

            # For now, repeat current action to create sequence
            actions_seq = actions.unsqueeze(1).repeat(1, config.action_horizon, 1)  # (B, T, A)

            # Forward pass
            batch_train = {"action": actions_seq.to(device)}
            loss, loss_dict = policy.forward(batch_train)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            # Logging
            epoch_loss += loss.item()
            epoch_recon_loss += loss_dict["recon_loss"]
            epoch_vq_loss += loss_dict["vq_loss"]

            if global_step % log_freq == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{loss_dict['recon_loss']:.4f}",
                    "vq": f"{loss_dict['vq_loss']:.4f}",
                })

            global_step += 1

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        avg_recon = epoch_recon_loss / len(dataloader)
        avg_vq = epoch_vq_loss / len(dataloader)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f}"
        )

        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": policy.vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = output_dir / "vae_final.pt"
    torch.save({
        "model_state_dict": policy.vae.state_dict(),
        "config": config,
    }, final_path)
    logger.info(f"Training complete! Final model saved: {final_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Train CARP VQ-VAE")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/carp_vae",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--save_freq", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log_freq", type=int, default=100,
                        help="Log every N steps")

    args = parser.parse_args()

    train_vae(
        dataset_path=args.dataset_path,
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
