"""
Artificial Muscle Memory Connection
====================================
A NumPy-based simulation of biological muscle memory using Hebbian
learning.  Motor-unit nodes are connected by a weight matrix whose
entries grow stronger every time two units fire together (practice)
and decay slowly over time (forgetting).
"""

from .network import MuscleMemoryNetwork
from .pattern import MotorPattern

__all__ = ["MuscleMemoryNetwork", "MotorPattern"]
__version__ = "0.1.0"
