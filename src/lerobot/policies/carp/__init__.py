"""CARP (Coarse-to-Fine Autoregressive Policy) for LeRobot."""

from .configuration_carp import CARPConfig

# Register CARP processor steps to ProcessorStepRegistry
# This must be imported to trigger the @ProcessorStepRegistry.register() decorators
from .carp_processor_steps import (
    CARPAddTemporalDimensionStep,
    CARPSampleActionSequenceStep,
)
from .carp_task_id_step import CARPTaskIDProcessorStep

__all__ = [
    "CARPConfig",
    "CARPAddTemporalDimensionStep",
    "CARPSampleActionSequenceStep",
    "CARPTaskIDProcessorStep",
]
