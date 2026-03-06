"""
Neural policy and muscle-memory agent.

NeuralPolicy
    A lightweight two-layer MLP implemented in pure NumPy.
    Trained with the REINFORCE (policy-gradient) algorithm.

MuscleMemoryAgent
    Wraps ``NeuralPolicy`` and ``MuscleMemory``.  On each decision step it
    first consults the muscle-memory cache; only if the context is not yet
    familiar does it invoke the neural network.  This models the shift from
    deliberate, computationally expensive reasoning to fast, automatic action
    execution as practice accumulates.
"""

from typing import Optional, Tuple

import numpy as np

from .memory import MuscleMemory


# ---------------------------------------------------------------------------
# Neural policy (pure NumPy two-layer MLP)
# ---------------------------------------------------------------------------


class NeuralPolicy:
    """
    Two-layer fully connected policy network with tanh hidden activation and
    softmax output.

    Parameters
    ----------
    input_size:
        Dimensionality of the observation vector.
    hidden_size:
        Number of hidden units.
    n_actions:
        Number of discrete actions (output logits).
    lr:
        Learning rate for the REINFORCE update.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_actions: int,
        lr: float = 0.01,
    ) -> None:
        self.n_actions = n_actions
        self.lr = lr

        # Xavier initialisation
        scale1 = np.sqrt(2.0 / input_size)
        scale2 = np.sqrt(2.0 / hidden_size)

        rng = np.random.default_rng(0)
        self.W1: np.ndarray = rng.standard_normal((input_size, hidden_size)) * scale1
        self.b1: np.ndarray = np.zeros(hidden_size)
        self.W2: np.ndarray = rng.standard_normal((hidden_size, n_actions)) * scale2
        self.b2: np.ndarray = np.zeros(n_actions)

        # Cached activations for the last forward pass (used during backprop)
        self._last_state: Optional[np.ndarray] = None
        self._last_hidden: Optional[np.ndarray] = None

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Compute action probabilities for ``state``.

        Returns
        -------
        np.ndarray
            Probability distribution over actions (sums to 1).
        """
        self._last_state = state
        hidden = np.tanh(state @ self.W1 + self.b1)
        self._last_hidden = hidden
        logits = hidden @ self.W2 + self.b2
        # Numerically stable softmax
        exp_l = np.exp(logits - logits.max())
        return exp_l / exp_l.sum()

    def update(self, state: np.ndarray, action: int, advantage: float) -> None:
        """
        Apply a single REINFORCE gradient update.

        Parameters
        ----------
        state:
            Observation at the time the action was taken.
        action:
            Index of the chosen action.
        advantage:
            Scalar advantage / return signal (e.g. discounted reward).
        """
        probs = self.forward(state)

        # Policy gradient: ∇log π(a|s) · advantage
        d_logits = probs.copy()
        d_logits[action] -= 1.0
        d_logits *= advantage * self.lr

        hidden = self._last_hidden  # type: ignore[assignment]

        # Gradients for W2, b2
        dW2 = np.outer(hidden, d_logits)
        db2 = d_logits

        # Backprop through tanh: d/dx tanh(x) = 1 − tanh²(x)
        d_hidden = (d_logits @ self.W2.T) * (1.0 - hidden ** 2)
        dW1 = np.outer(state, d_hidden)
        db1 = d_hidden

        # Gradient descent
        self.W1 -= dW1
        self.b1 -= db1
        self.W2 -= dW2
        self.b2 -= db2


# ---------------------------------------------------------------------------
# Muscle-memory agent
# ---------------------------------------------------------------------------


class MuscleMemoryAgent:
    """
    Reinforcement learning agent with an artificial muscle memory cache.

    Decision logic per step
    -----------------------
    1. Query ``MuscleMemory`` with the current context key.
    2. If a cached action exists → return it instantly (fast path, no NN).
    3. Otherwise → run ``NeuralPolicy.forward`` (slow path, NN inference).

    As training progresses, more contexts become familiar and an increasing
    fraction of decisions are served from the cache, reducing computational
    overhead — exactly as in biological muscle memory.

    Parameters
    ----------
    state_size:
        Dimensionality of the observation from ``SequenceEnvironment``.
    n_actions:
        Number of discrete actions.
    hidden_size:
        Hidden units in the neural policy.
    lr:
        Learning rate for the REINFORCE update.
    familiarity_threshold:
        Correct executions required to cache an action.
    epsilon:
        Probability of random exploration when the neural network is used.
    """

    def __init__(
        self,
        state_size: int,
        n_actions: int,
        hidden_size: int = 32,
        lr: float = 0.01,
        familiarity_threshold: int = 10,
        epsilon: float = 0.1,
    ) -> None:
        self.n_actions = n_actions
        self.epsilon = epsilon

        self.policy = NeuralPolicy(state_size, hidden_size, n_actions, lr)
        self.memory = MuscleMemory(familiarity_threshold)

        # Running totals — used to measure efficiency
        self.nn_calls: int = 0
        self.cache_hits: int = 0
        self.total_calls: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def act(
        self,
        state: np.ndarray,
        context: Tuple,
        explore: bool = True,
    ) -> Tuple[int, bool]:
        """
        Select an action for the given state and context.

        Parameters
        ----------
        state:
            Current environment observation.
        context:
            Hashable context key (from ``SequenceEnvironment.context_key()``).
        explore:
            When ``True`` epsilon-greedy exploration is active.

        Returns
        -------
        (action, from_cache)
            ``from_cache`` is ``True`` when the action was served by the
            muscle memory cache (no neural network inference performed).
        """
        self.total_calls += 1

        # --- Fast path: muscle memory cache ---
        cached = self.memory.lookup(context)
        if cached is not None:
            self.cache_hits += 1
            return cached, True

        # --- Slow path: neural network ---
        self.nn_calls += 1
        if explore and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions)), False

        probs = self.policy.forward(state)
        action = int(np.random.choice(self.n_actions, p=probs))
        return action, False

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        context: Tuple,
        success: bool,
    ) -> None:
        """
        Update both the neural policy and the muscle memory.

        Parameters
        ----------
        state:
            Observation at the time of the action.
        action:
            Action that was taken.
        reward:
            Scalar reward signal.
        context:
            Context key at the time of the action.
        success:
            Whether the action was correct (used for cache bookkeeping).
        """
        self.policy.update(state, action, reward)
        self.memory.update(context, action, success)

    # ------------------------------------------------------------------
    # Efficiency statistics
    # ------------------------------------------------------------------

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of all decisions served from the muscle memory cache."""
        if self.total_calls == 0:
            return 0.0
        return self.cache_hits / self.total_calls

    @property
    def efficiency_gain(self) -> float:
        """
        Proportion of neural-network inference saved thanks to caching.

        Equivalent to ``cache_hit_rate``: a value of 0.8 means 80 % of
        decisions required no NN computation.
        """
        return self.cache_hit_rate
