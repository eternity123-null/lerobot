"""Custom processor step for CARP task ID handling."""

from dataclasses import dataclass

import torch

from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import EnvTransition
from lerobot.configs.types import PolicyFeature


@ProcessorStepRegistry.register(name="carp_task_id")
@dataclass
class CARPTaskIDProcessorStep(ProcessorStep):
    """
    Handle task ID for CARP multi-task policy.

    Renames 'task_index' (from dataset) to 'task_id' (expected by model).
    Preserves existing 'task_id' if already present.

    Task_index is stored in transition["complementary_data"]["task_index"]
    after batch_to_transition conversion.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Rename task_index to task_id in complementary_data.

        LIBERO dataset provides: transition["complementary_data"]["task_index"]
        CARP model expects: transition["complementary_data"]["task_id"]
        """
        comp_data = transition.get("complementary_data", {})

        # Check if task_index exists in complementary_data and rename it
        if "task_index" in comp_data and comp_data["task_index"] is not None:
            comp_data["task_id"] = comp_data.pop("task_index")

        # Check top-level as well (for environment case)
        if "task_index" in transition and transition["task_index"] is not None:
            comp_data["task_id"] = transition.pop("task_index")

        # If task_id already exists (e.g., from environment), keep it
        if "task_id" in transition and transition["task_id"] is not None:
            comp_data["task_id"] = transition.pop("task_id")

        # Update complementary_data
        if comp_data:
            transition["complementary_data"] = comp_data

        return transition

    def transform_features(
        self, features: dict[str, dict[str, PolicyFeature]]
    ) -> dict[str, dict[str, PolicyFeature]]:
        """
        This step doesn't change feature shapes, just renames a field.
        """
        return features
