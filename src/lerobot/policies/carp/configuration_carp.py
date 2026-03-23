#!/usr/bin/env python

# Copyright 2026 CARP Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration class for CARP (Coarse-to-Fine Autoregressive Policy)."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig


@PreTrainedConfig.register_subclass("carp")
@dataclass
class CARPConfig(PreTrainedConfig):
    """
    Configuration class for CARP (Coarse-to-Fine Autoregressive Policy).

    CARP is a two-stage training policy:
    1. MSAT (Multi-Scale Action Tokenization): Train VQ-VAE tokenizer
    2. CFAP (Coarse-to-Fine Autoregressive Prediction): Train Transformer

    Args:
        n_obs_steps: Number of observation steps to use as context.
        action_horizon: Length of action sequence to predict.

        # MSAT (VQ-VAE) parameters
        vocab_size: Size of the codebook for vector quantization.
        vocab_ch: Latent dimension (z_channels) for the VAE.
        vch: Base number of channels in VAE encoder/decoder.
        ch_mult: Channel multipliers for downsampling.
        vae_dropout: Dropout rate in VAE.
        vqbeta: Commitment loss weight for VQ.
        vqnorm: Use cosine similarity (True) vs Euclidean distance (False) for quantization.
        vqresi: Residual connection ratio in quantizer.
        vae_lr: Learning rate for VAE training.
        vae_weight_decay: Weight decay for VAE training.
        vae_warmup_ratio: Warmup ratio for VAE training.
        vae_lr_schedule: Learning rate schedule for VAE ("cos", "lin", "lin0").

        # CFAP (Transformer) parameters
        ar_depth: Number of transformer layers.
        ar_embed_dim: Embedding dimension.
        ar_num_heads: Number of attention heads.
        ar_mlp_ratio: MLP hidden dim = ar_embed_dim * ar_mlp_ratio.
        ar_dropout: Dropout rate in transformer.
        ar_attn_dropout: Dropout rate in attention.
        ar_drop_path_rate: Drop path rate for stochastic depth.
        ar_lr: Learning rate for AR training.
        ar_weight_decay: Weight decay for AR training.
        ar_warmup_ratio: Warmup ratio for AR training.
        ar_lr_schedule: Learning rate schedule for AR ("cos", "lin", "lin0").
        ar_label_smoothing: Label smoothing for cross-entropy loss.

        # Multi-scale configuration
        patch_nums: Number of patches at each scale (e.g., (1, 2, 3, 4) for 4 scales).
        patch_size: Size of each patch (usually 1 for actions).

        # Multi-task configuration
        task_num: Number of tasks for multi-task learning.
        task_embed_dim: Embedding dimension for task IDs.

        # Training stage control
        training_stage: "vae" or "ar" to control which stage to train.
        vae_checkpoint_path: Path to pretrained VAE checkpoint (required for AR stage).

        # Normalization
        normalization_mapping: Dict mapping feature types to normalization modes.
    """

    # ============ General configuration ============
    n_obs_steps: int = 1
    action_horizon: int = 16
    n_action_steps: int = 16 # Number of action steps to execute (can be <= action_horizon)

    # ============ MSAT (VQ-VAE) configuration ============
    vocab_size: int = 1024  # Codebook size (original CARP uses 1024)
    vocab_ch: int = 8
    vch: int = 2
    ch_mult: tuple[int, ...] = (2, 4)
    vae_dropout: float = 0.0
    vqbeta: float = 0.25
    vqnorm: bool = True
    vqresi: float = 0.5
    vae_lr: float = 3e-4
    vae_weight_decay: float = 0.005
    vae_warmup_ratio: float = 0.0
    vae_lr_schedule: str = "cos"

    # ============ CFAP (Transformer) configuration ============
    ar_depth: int = 32
    ar_embed_dim: int = 160
    ar_num_heads: int = 32
    ar_mlp_ratio: float = 4.0
    ar_dropout: float = 0.0
    ar_attn_dropout: float = 0.0
    ar_drop_path_rate: float = 0.0
    ar_lr: float = 1e-4
    ar_weight_decay: float = 0.05
    ar_warmup_ratio: float = 0.0
    ar_lr_schedule: str = "lin0"
    ar_label_smoothing: float = 0.0

    # ============ Multi-scale configuration ============
    patch_nums: tuple[int, ...] = (1, 2, 3, 4)
    patch_size: int = 1

    # ============ Multi-task configuration ============
    task_num: int = 40
    task_embed_dim: int = 3

    # ============ Training stage control ============
    training_stage: str = "ar"  # "vae" or "ar"
    vae_checkpoint_path: str | None = None

    # ============ Normalization configuration ============
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,  # Images not normalized
            "STATE": NormalizationMode.MEAN_STD,  # State normalized
            "ACTION": NormalizationMode.MEAN_STD,  # Action normalized
        }
    )

    def get_optimizer_preset(self) -> AdamWConfig:
        """Return optimizer config based on training stage."""
        if self.training_stage == "vae":
            return AdamWConfig(
                lr=self.vae_lr,
                weight_decay=self.vae_weight_decay,
                betas=[0.5, 0.9],
            )
        else:  # ar
            return AdamWConfig(
                lr=self.ar_lr,
                weight_decay=self.ar_weight_decay,
                betas=[0.9, 0.999],
            )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """Return learning rate scheduler config based on training stage."""
        from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

        if self.training_stage == "vae":
            schedule = self.vae_lr_schedule
            warmup = self.vae_warmup_ratio
        else:
            schedule = self.ar_lr_schedule
            warmup = self.ar_warmup_ratio

        if schedule == "cos":
            return CosineDecayWithWarmupSchedulerConfig(
                num_warmup_steps=warmup,
                num_decay_steps=10000,
                peak_lr=0.0001,
                decay_lr=0.00001,
            )
        return None

    def validate_features(self) -> None:
        """Validate input/output features."""
        assert self.input_features is not None, "input_features must be set"
        assert self.output_features is not None, "output_features must be set"

        # Validate action feature
        action_ft = self.output_features.get("action")
        assert action_ft is not None, "action feature is required"

        # For AR stage, validate observation features
        if self.training_stage == "ar":
            has_images = any(k.startswith("observation.images.") for k in self.input_features)
            has_state = "observation.state" in self.input_features
            assert has_images or has_state, "AR stage needs at least images or state"

    @property
    def observation_delta_indices(self) -> None:
        """CARP does not use delta observations."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """CARP uses delta actions - returns indices for action_horizon steps."""
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        """CARP does not use delta rewards."""
        return None


@PreTrainedConfig.register_subclass("carp_vae")
@dataclass
class CARPVAEConfig(CARPConfig):
    """
    Configuration class for CARP VAE (Stage 1: Multi-Scale Action Tokenization).

    This is a simplified config that forces training_stage="vae" and removes AR-specific params.
    """

    training_stage: str = "vae"  # Fixed to VAE stage

    def __post_init__(self):
        super().__post_init__()
        # Force VAE stage
        self.training_stage = "vae"
