#!/usr/bin/env python

"""Data processors for CARP policy."""

from typing import Any

import torch

from lerobot.policies.carp.configuration_carp import CARPConfig
from lerobot.policies.carp.carp_processor_steps import (
    CARPSampleActionSequenceStep,
    CARPAddTemporalDimensionStep,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def make_carp_pre_post_processors(
    config: CARPConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],  # Preprocessor
    PolicyProcessorPipeline[torch.Tensor, torch.Tensor],       # Postprocessor
]:
    """
    Create preprocessor and postprocessor pipelines for CARP policy.

    Preprocessor (Robot/Env → Policy):
        1. RenameObservations - ensure consistent naming
        2. AddBatchDimension - add batch dim if needed
        3. Normalizer - normalize observations and actions
        4. SampleActionSequence - sample action_horizon steps (for training)
        5. AddTemporalDimension - add time dimension to observations
        6. Device - move to GPU/CPU

    Postprocessor (Policy → Robot/Env):
        1. Unnormalizer - denormalize actions
        2. Device - move to CPU

    Args:
        config: CARP configuration
        dataset_stats: Dataset statistics for normalization

    Returns:
        (preprocessor, postprocessor) pipelines
    """

    # ========== Preprocessor ==========
    input_steps = []

    # Step 1: Rename observations (if needed)
    rename_map = {}  # Can be customized based on robot/env naming conventions
    if rename_map:
        input_steps.append(RenameObservationsProcessorStep(rename_map=rename_map))

    # Step 2: Add batch dimension (for single samples)
    input_steps.append(AddBatchDimensionProcessorStep())

    # Step 3: Normalize observations and actions
    # IMPORTANT: This must come before any tokenizer steps (for VLA models)
    if dataset_stats is not None:
        input_steps.append(
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            )
        )

    # Step 4: Sample action sequences (for training)
    # Convert action: (B, A) -> (B, action_horizon, A)
    input_steps.append(
        CARPSampleActionSequenceStep(action_horizon=config.action_horizon)
    )

    # Step 5: Add temporal dimension to observations
    # Convert obs: (B, ...) -> (B, n_obs_steps, ...)
    # ALWAYS add this step, even if n_obs_steps=1, because AR model expects it
    input_steps.append(
        CARPAddTemporalDimensionStep(n_obs_steps=config.n_obs_steps)
    )

    # Step 6: Move to target device
    input_steps.append(DeviceProcessorStep(device=config.device))

    # ========== Postprocessor ==========
    output_steps = []

    # Step 1: Unnormalize actions
    if dataset_stats is not None:
        output_steps.append(
            UnnormalizerProcessorStep(
                features=config.output_features,
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            )
        )

    # Step 2: Move to CPU (for robot execution)
    output_steps.append(DeviceProcessorStep(device="cpu"))

    # ========== Create pipelines ==========
    preprocessor = PolicyProcessorPipeline(
        steps=input_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )

    postprocessor = PolicyProcessorPipeline(
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    return preprocessor, postprocessor
