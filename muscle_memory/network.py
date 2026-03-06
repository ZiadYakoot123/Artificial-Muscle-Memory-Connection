"""Core Muscle-Memory Network implementing Hebbian learning.

Biological background
---------------------
Human muscle memory (procedural memory) is encoded in the cerebellum
and motor cortex through repeated practice.  At the synaptic level the
"Hebbian rule" applies: *neurons that fire together, wire together*.
The connection weight between two motor units increases every time both
are co-activated during a practice trial and decays when the skill is
not exercised.

Implementation
--------------
* ``n_units`` motor-unit nodes, fully connected via a weight matrix W
  of shape (n_units, n_units).
* **Practice** – for each consecutive pair of activation vectors in a
  :class:`~muscle_memory.pattern.MotorPattern` the outer product is
  added to W (Hebbian update), clipped to [0, 1].
* **Recall** – given an initial activation vector the network
  propagates it through W step-by-step (with a sigmoid non-linearity)
  to reconstruct the rest of the pattern.
* **Forgetting** – a decay step reduces all weights by a configurable
  rate, simulating synaptic pruning.
"""

from __future__ import annotations

import numpy as np

from .pattern import MotorPattern


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class MuscleMemoryNetwork:
    """A Hebbian network that learns motor patterns through practice.

    Parameters
    ----------
    n_units:
        Number of motor-unit nodes.
    learning_rate:
        Scaling factor applied to each Hebbian weight update.
    decay_rate:
        Fraction by which all weights decrease on each call to
        :meth:`forget`.  Must be in [0, 1].
    rng:
        Optional NumPy random generator used to initialise the weight
        matrix and to break ties during recall.
    """

    def __init__(
        self,
        n_units: int,
        learning_rate: float = 0.1,
        decay_rate: float = 0.01,
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_units < 1:
            raise ValueError("n_units must be >= 1")
        if not (0 < learning_rate <= 1):
            raise ValueError("learning_rate must be in (0, 1]")
        if not (0 <= decay_rate < 1):
            raise ValueError("decay_rate must be in [0, 1)")

        self._n_units = n_units
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self._rng = rng or np.random.default_rng()

        # Weight matrix – small random initial values
        self._W: np.ndarray = (
            self._rng.random((n_units, n_units)) * 0.01
        )
        np.fill_diagonal(self._W, 0.0)  # no self-connections

        # Number of times each pattern has been practised
        self._practice_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_units(self) -> int:
        return self._n_units

    @property
    def weights(self) -> np.ndarray:
        """Read-only copy of the current weight matrix."""
        return self._W.copy()

    @property
    def practice_counts(self) -> dict[str, int]:
        """Mapping of pattern name → number of practice trials."""
        return dict(self._practice_counts)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def practice(self, pattern: MotorPattern, n_trials: int = 1) -> None:
        """Strengthen synaptic connections by practising *pattern*.

        For each consecutive pair of activation vectors ``(a_t, a_{t+1})``
        in the pattern the weight matrix is updated using the outer
        product (Hebbian rule):

        .. math::
            W \\mathrel{+}= \\eta \\cdot a_t \\otimes a_{t+1}

        The diagonal (self-connections) is kept at zero.  All weights
        are clipped to [0, 1] after the update.

        Parameters
        ----------
        pattern:
            The motor pattern to rehearse.
        n_trials:
            How many times to repeat the practice session.
        """
        if pattern.n_units != self._n_units:
            raise ValueError(
                f"Pattern has {pattern.n_units} units but network has "
                f"{self._n_units}"
            )
        acts = pattern.activations
        for _ in range(n_trials):
            for t in range(acts.shape[0] - 1):
                delta = np.outer(acts[t], acts[t + 1])
                self._W += self.learning_rate * delta
            np.fill_diagonal(self._W, 0.0)
            np.clip(self._W, 0.0, 1.0, out=self._W)

        key = pattern.name or id(pattern)
        self._practice_counts[str(key)] = (
            self._practice_counts.get(str(key), 0) + n_trials
        )

    def recall(
        self, initial_activation: np.ndarray, n_steps: int
    ) -> MotorPattern:
        """Reconstruct *n_steps* of a motor pattern from a cue.

        Starting from *initial_activation* the network propagates
        activity through the weight matrix at each step:

        .. math::
            a_{t+1} = \\sigma(W^\\top \\cdot a_t)

        where :math:`\\sigma` is the logistic sigmoid.

        Parameters
        ----------
        initial_activation:
            1-D array of length ``n_units`` in [0, 1].
        n_steps:
            Number of additional steps to generate (the returned
            pattern will have ``n_steps + 1`` rows including the cue).

        Returns
        -------
        MotorPattern
            The recalled sequence starting with *initial_activation*.
        """
        cue = np.asarray(initial_activation, dtype=float)
        if cue.shape != (self._n_units,):
            raise ValueError(
                f"initial_activation must have shape ({self._n_units},); "
                f"got {cue.shape}"
            )
        rows = [cue.copy()]
        current = cue.copy()
        for _ in range(n_steps):
            current = _sigmoid(self._W.T @ current)
            rows.append(current.copy())
        return MotorPattern(np.vstack(rows), name="recalled")

    def forget(self, n_steps: int = 1) -> None:
        """Apply synaptic decay to simulate forgetting.

        .. math::
            W \\mathrel{*}= (1 - \\text{decay\\_rate})^{\\text{n\\_steps}}

        Parameters
        ----------
        n_steps:
            Number of forgetting steps to apply at once.
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        factor = (1.0 - self.decay_rate) ** n_steps
        self._W *= factor

    def connection_strength(self, unit_a: int, unit_b: int) -> float:
        """Return the current weight of the connection *unit_a* → *unit_b*.

        Parameters
        ----------
        unit_a:
            Index of the source motor unit.
        unit_b:
            Index of the target motor unit.
        """
        return float(self._W[unit_a, unit_b])

    def strongest_connections(self, top_k: int = 5) -> list[tuple[int, int, float]]:
        """Return the *top_k* strongest directed connections.

        Returns
        -------
        list of (unit_a, unit_b, weight)
            Sorted in descending order of weight.
        """
        flat_indices = np.argsort(self._W.ravel())[::-1][:top_k]
        results = []
        for idx in flat_indices:
            i, j = divmod(int(idx), self._n_units)
            results.append((i, j, float(self._W[i, j])))
        return results

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MuscleMemoryNetwork("
            f"n_units={self._n_units}, "
            f"learning_rate={self.learning_rate}, "
            f"decay_rate={self.decay_rate})"
        )
