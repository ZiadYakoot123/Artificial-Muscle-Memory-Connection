"""
Sequence reproduction environment for the artificial muscle memory simulation.

The agent must reproduce target sequences of discrete actions, analogous to
motor tasks where repeated practice builds automaticity (e.g. typing patterns,
musical phrases, athletic movements).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class SequenceEnvironment:
    """
    A reinforcement learning environment in which the agent must reproduce
    a target sequence of actions one step at a time.

    Observation space
    -----------------
    A 1-D float array of size ``n_actions + 2``:

    * index 0   : normalised progress through the current sequence [0, 1)
    * index 1…n : one-hot encoding of the *previous* action taken
    * index -1  : normalised sequence index (useful when multiple sequences
                  are used so the agent can identify which pattern to recall)

    Action space
    ------------
    Integers in ``[0, n_actions)``.

    Reward
    ------
    * ``+1.0`` per correct step
    * ``+10.0`` bonus on successful sequence completion
    * ``-1.0`` for an incorrect step (episode ends)
    """

    def __init__(self, sequences: List[List[int]], n_actions: int) -> None:
        if not sequences:
            raise ValueError("At least one sequence must be provided.")
        if n_actions < 1:
            raise ValueError("n_actions must be >= 1.")
        for seq in sequences:
            if not seq:
                raise ValueError("Each sequence must contain at least one action.")
            if any(a < 0 or a >= n_actions for a in seq):
                raise ValueError(
                    f"All actions must be in [0, {n_actions}). Got: {seq}"
                )

        self.sequences: List[List[int]] = [list(s) for s in sequences]
        self.n_actions: int = n_actions
        # obs = progress + prev_action one-hot + sequence id
        self.state_size: int = n_actions + 2

        self._seq_idx: int = 0
        self.current_sequence: Optional[List[int]] = None
        self.current_step: int = 0

    # ------------------------------------------------------------------
    # Core gym-style interface
    # ------------------------------------------------------------------

    def reset(self, seq_idx: Optional[int] = None) -> np.ndarray:
        """
        Start a new episode.

        Parameters
        ----------
        seq_idx:
            If given, select this specific sequence (mod ``len(sequences)``).
            Otherwise a sequence is chosen uniformly at random.

        Returns
        -------
        np.ndarray
            Initial observation.
        """
        if seq_idx is not None:
            self._seq_idx = int(seq_idx) % len(self.sequences)
        else:
            self._seq_idx = int(np.random.randint(len(self.sequences)))
        self.current_sequence = self.sequences[self._seq_idx]
        self.current_step = 0
        return self._observe()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action:
            Discrete action chosen by the agent.

        Returns
        -------
        (observation, reward, done, info)
        """
        if self.current_sequence is None:
            raise RuntimeError("Call reset() before step().")

        target = self.current_sequence[self.current_step]
        correct = int(action) == int(target)

        if correct:
            self.current_step += 1
            done = self.current_step >= len(self.current_sequence)
            reward = 10.0 if done else 1.0
        else:
            done = True
            reward = -1.0

        obs = self._observe() if not done else np.zeros(self.state_size)
        info = {
            "correct": correct,
            "step": self.current_step,
            "seq_len": len(self.current_sequence),
            "seq_idx": self._seq_idx,
        }
        return obs, reward, done, info

    def context_key(self) -> Tuple:
        """
        Return a hashable key that uniquely identifies the current position
        within the current sequence — used as the muscle-memory lookup key.
        """
        return (self._seq_idx, self.current_step)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        obs = np.zeros(self.state_size)
        seq_len = len(self.current_sequence)  # type: ignore[arg-type]
        obs[0] = self.current_step / seq_len
        if self.current_step > 0:
            prev_action = self.current_sequence[self.current_step - 1]  # type: ignore[index]
            obs[1 + prev_action] = 1.0
        n_seq = len(self.sequences)
        obs[-1] = float(self._seq_idx) / max(1, n_seq - 1)
        return obs
