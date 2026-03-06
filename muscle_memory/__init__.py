"""
Artificial Muscle Memory — reinforcement learning simulation.

This package implements a neural network agent that learns to execute
repeated action sequences with increasing efficiency, modelling the
concept of muscle memory in biological motor learning.
"""

from .environment import SequenceEnvironment
from .memory import MuscleMemory
from .agent import MuscleMemoryAgent, NeuralPolicy
from .trainer import Trainer, EpisodeStats

__all__ = [
    "SequenceEnvironment",
    "MuscleMemory",
    "MuscleMemoryAgent",
    "NeuralPolicy",
    "Trainer",
    "EpisodeStats",
]
