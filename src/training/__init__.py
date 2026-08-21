"""Training module for Vi-VQA"""

from .config_utils import ConfigLoader
from .trainer_utils import (
    VQATrainerCallback,
    MetricsCalculator,
    TrainingHelper,
    CheckpointManager,
)
try:
    from .training_loop import VQATrainer, DistillationTrainer
except ImportError:
    # Evaluation/inference can run with the lightweight runtime dependencies.
    VQATrainer = None
    DistillationTrainer = None

__all__ = [
    "ConfigLoader",
    "VQATrainerCallback",
    "MetricsCalculator",
    "TrainingHelper",
    "CheckpointManager",
    "VQATrainer",
    "DistillationTrainer",
]
