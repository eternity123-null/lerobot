"""Observation encoder builder for CARP policy."""

import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.carp.configuration_carp import CARPConfig


class CARPObsEncoder(nn.Module):
    """
    Observation encoder for CARP policy.

    Encodes multi-modal observations (images + state) into a fixed-dim feature vector.

    Architecture:
        - Images → ResNet18 → [128 per camera]
        - State → Identity/MLP → [D]
        - Output: Concatenated features [obs_dim]
    """

    def __init__(self, config: CARPConfig):
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleDict()
        self.output_dims = []

        # Build image encoders
        self.image_keys = [k for k in config.input_features if k.startswith("observation.images.")]
        for img_key in self.image_keys:
            img_shape = config.input_features[img_key].shape  # (C, H, W)
            # Use ResNet18 as backbone
            backbone_model = torchvision.models.resnet18(
                weights="IMAGENET1K_V1",
                norm_layer=FrozenBatchNorm2d,
            )
            # Extract features from layer4 (final layer)
            backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

            # Add adaptive pooling and projection to get fixed size output
            fc_in_features = backbone_model.fc.in_features  # 512 for ResNet18
            image_encoder = nn.Sequential(
                backbone,
                nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
                nn.Flatten(),
                nn.Linear(fc_in_features, 128),  # Project to 128-dim
            )

            # Wrap in a module to extract from dict
            class ImageEncoderWrapper(nn.Module):
                def __init__(self, encoder):
                    super().__init__()
                    self.encoder = encoder

                def forward(self, x):
                    # x shape: (B*T, C, H, W) or (B, C, H, W)
                    out = self.encoder[0](x)  # backbone returns dict
                    feat = out["feature_map"]  # (B, 512, H', W')
                    feat = self.encoder[1](feat)  # adaptive pool -> (B, 512, 1, 1)
                    feat = self.encoder[2](feat)  # flatten -> (B, 512)
                    feat = self.encoder[3](feat)  # linear -> (B, 128)
                    return feat

            # Use a sanitized key (replace dots with underscores)
            sanitized_key = img_key.replace(".", "_")
            self.encoders[sanitized_key] = ImageEncoderWrapper(image_encoder)
            self.output_dims.append(128)

        # Build state encoder (identity mapping)
        self.has_state = "observation.state" in config.input_features
        if self.has_state:
            state_dim = config.input_features["observation.state"].shape[0]
            self.encoders["observation_state"] = nn.Identity()
            self.output_dims.append(state_dim)

        self._output_dim = sum(self.output_dims)

    def forward(self, obs_dict):
        """
        Args:
            obs_dict: Dict of observations
                - "observation.images.*": (B, T, C, H, W) or (B, C, H, W)
                - "observation.state": (B, T, D) or (B, D)

        Returns:
            features: (B, obs_dim) or (B, T, obs_dim)
        """
        features = []

        # Process images
        for img_key in self.image_keys:
            if img_key in obs_dict:
                img = obs_dict[img_key]  # (B, T, C, H, W) or (B, C, H, W)

                # Get sanitized key for encoder lookup
                sanitized_key = img_key.replace(".", "_")

                # Handle temporal dimension
                if img.ndim == 5:  # (B, T, C, H, W)
                    B, T, C, H, W = img.shape
                    img = img.reshape(B * T, C, H, W)
                    feat = self.encoders[sanitized_key](img)  # (B*T, 128)
                    feat = feat.reshape(B, T, -1)  # (B, T, 128)
                else:  # (B, C, H, W)
                    feat = self.encoders[sanitized_key](img)  # (B, 128)

                features.append(feat)

        # Process state
        if self.has_state and "observation.state" in obs_dict:
            state = obs_dict["observation.state"]  # (B, T, D) or (B, D)
            feat = self.encoders["observation_state"](state)
            features.append(feat)

        # Concatenate all features
        if features:
            return torch.cat(features, dim=-1)
        else:
            raise ValueError("No observations to encode")

    def output_shape(self):
        """Return output shape as [obs_dim]."""
        return [self._output_dim]


def build_carp_obs_encoder(config: CARPConfig) -> CARPObsEncoder:
    """
    Build CARP observation encoder.

    Args:
        config: CARP configuration

    Returns:
        CARPObsEncoder instance
    """
    return CARPObsEncoder(config)
