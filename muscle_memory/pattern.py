"""Motor pattern representation for the muscle-memory network."""

from __future__ import annotations

import numpy as np


class MotorPattern:
    """A sequence of motor-unit activation vectors.

    Each step in the sequence is a 1-D NumPy array of length
    ``n_units`` whose values lie in [0, 1].  Values close to 1 mean
    the corresponding motor unit is strongly activated at that step.

    Parameters
    ----------
    activations:
        2-D array of shape (n_steps, n_units).  Each row is one
        time-step of the movement.
    name:
        Optional human-readable label for the pattern.
    """

    def __init__(self, activations: np.ndarray, name: str = "") -> None:
        activations = np.asarray(activations, dtype=float)
        if activations.ndim != 2:
            raise ValueError(
                "activations must be 2-D (n_steps × n_units); "
                f"got shape {activations.shape}"
            )
        if activations.min() < 0 or activations.max() > 1:
            raise ValueError("activation values must be in [0, 1]")
        self._activations = activations.copy()
        self.name = name

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def activations(self) -> np.ndarray:
        """Read-only view of the activation array."""
        return self._activations

    @property
    def n_steps(self) -> int:
        """Number of time-steps in the pattern."""
        return self._activations.shape[0]

    @property
    def n_units(self) -> int:
        """Number of motor units the pattern spans."""
        return self._activations.shape[1]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        tag = f" '{self.name}'" if self.name else ""
        return (
            f"MotorPattern{tag}("
            f"n_steps={self.n_steps}, n_units={self.n_units})"
        )

    @classmethod
    def random(
        cls,
        n_steps: int,
        n_units: int,
        sparsity: float = 0.5,
        rng: np.random.Generator | None = None,
        name: str = "",
    ) -> "MotorPattern":
        """Create a random sparse motor pattern.

        Parameters
        ----------
        n_steps:
            Number of time-steps.
        n_units:
            Number of motor units.
        sparsity:
            Fraction of units that are *silent* (0) at each step.
            E.g. 0.7 means 70 % of units are inactive at each step.
        rng:
            Optional NumPy random generator for reproducibility.
        name:
            Optional label.
        """
        if rng is None:
            rng = np.random.default_rng()
        raw = rng.random((n_steps, n_units))
        mask = rng.random((n_steps, n_units)) < sparsity
        raw[mask] = 0.0
        return cls(raw, name=name)
