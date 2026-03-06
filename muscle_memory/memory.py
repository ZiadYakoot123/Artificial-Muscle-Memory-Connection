"""
Muscle memory module — familiarity-based action cache.

Biological muscle memory works by transferring frequently practised
motor programmes from conscious deliberation to automatic execution.
This module simulates that mechanism:

* Every time the agent *correctly* executes an action in a given context,
  a familiarity counter for that (context, action) pair is incremented.
* When the counter reaches ``familiarity_threshold``, the action is
  promoted to the cache and can be retrieved instantly — bypassing neural
  network inference entirely.
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple


class MuscleMemory:
    """
    Familiarity-based action cache.

    Parameters
    ----------
    familiarity_threshold:
        Number of correct executions required before an action is cached
        (i.e. considered "automatic").
    """

    def __init__(self, familiarity_threshold: int = 10) -> None:
        if familiarity_threshold < 1:
            raise ValueError("familiarity_threshold must be >= 1.")
        self.familiarity_threshold: int = familiarity_threshold

        # Maps context → {action: correct_execution_count}
        self._counts: Dict[Tuple, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Maps context → cached action (fully automatic)
        self._cache: Dict[Tuple, int] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, context: Tuple, action: int, success: bool) -> None:
        """
        Record the outcome of executing ``action`` in ``context``.

        Parameters
        ----------
        context:
            Hashable environment context (e.g. ``(seq_idx, step)``).
        action:
            The action that was taken.
        success:
            Whether the action was correct / rewarded.
        """
        if success:
            self._counts[context][action] += 1
            if self._counts[context][action] >= self.familiarity_threshold:
                self._cache[context] = action

    def lookup(self, context: Tuple) -> Optional[int]:
        """
        Return the cached action for ``context`` if familiar, else ``None``.

        A ``None`` return means the agent must fall back to neural inference.
        """
        return self._cache.get(context)

    def familiarity_count(self, context: Tuple, action: int) -> int:
        """Return the number of correct executions recorded for (context, action)."""
        return self._counts[context][action]

    def is_cached(self, context: Tuple) -> bool:
        """Return ``True`` if the context has a fully cached action."""
        return context in self._cache

    # ------------------------------------------------------------------
    # Properties / statistics
    # ------------------------------------------------------------------

    @property
    def n_cached(self) -> int:
        """Number of context positions whose action is fully cached."""
        return len(self._cache)

    @property
    def n_tracked(self) -> int:
        """Number of context positions that have been seen at least once."""
        return len(self._counts)

    def familiarity_ratio(self) -> float:
        """
        Fraction of tracked contexts that have been fully cached.

        Returns 0.0 when nothing has been tracked yet.
        """
        if self.n_tracked == 0:
            return 0.0
        return self.n_cached / self.n_tracked
