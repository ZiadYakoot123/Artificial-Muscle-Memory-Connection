"""
Training loop and episode statistics for the artificial muscle memory simulation.
"""

import time
from typing import Any, Dict, List

import numpy as np

from .agent import MuscleMemoryAgent
from .environment import SequenceEnvironment


class EpisodeStats:
    """
    Accumulates per-episode metrics and provides windowed summary statistics.

    Attributes tracked per episode
    --------------------------------
    rewards     : total undiscounted return
    nn_calls    : number of neural-network forward passes
    cache_hits  : number of cache-served decisions
    success     : whether the full sequence was completed correctly
    durations   : wall-clock time for the episode (seconds)
    """

    def __init__(self) -> None:
        self.rewards: List[float] = []
        self.nn_calls: List[int] = []
        self.cache_hits: List[int] = []
        self.success: List[bool] = []
        self.durations: List[float] = []

    def record(
        self,
        reward: float,
        nn_calls: int,
        cache_hits: int,
        success: bool,
        duration: float,
    ) -> None:
        """Append one episode's metrics."""
        self.rewards.append(reward)
        self.nn_calls.append(nn_calls)
        self.cache_hits.append(cache_hits)
        self.success.append(success)
        self.durations.append(duration)

    def window_stats(self, window: int = 100) -> Dict[str, float]:
        """
        Return aggregated statistics over the last ``window`` episodes.
        """
        n = min(window, len(self.rewards))
        if n == 0:
            return {}
        return self._compute_stats(
            self.rewards[-n:],
            self.nn_calls[-n:],
            self.cache_hits[-n:],
            self.success[-n:],
        )

    def window_stats_range(self, start: int, end: int) -> Dict[str, float]:
        """
        Return aggregated statistics for episodes in ``[start, end)``.
        Indices are clamped to the available range.
        """
        start = max(0, start)
        end = min(len(self.rewards), end)
        if start >= end:
            return {}
        return self._compute_stats(
            self.rewards[start:end],
            self.nn_calls[start:end],
            self.cache_hits[start:end],
            self.success[start:end],
        )

    @staticmethod
    def _compute_stats(
        rewards: List[float],
        nn_calls: List[int],
        cache_hits: List[int],
        success: List[bool],
    ) -> Dict[str, float]:
        total_decisions = np.mean(nn_calls) + np.mean(cache_hits)
        cache_rate = (
            np.mean(cache_hits) / total_decisions if total_decisions > 0 else 0.0
        )
        return {
            "mean_reward": float(np.mean(rewards)),
            "success_rate": float(np.mean(success)),
            "mean_nn_calls": float(np.mean(nn_calls)),
            "mean_cache_hits": float(np.mean(cache_hits)),
            "cache_hit_rate": float(cache_rate),
        }

    def __len__(self) -> int:
        return len(self.rewards)


class Trainer:
    """
    Runs the training loop, accumulates ``EpisodeStats``, and prints
    periodic progress reports showing the efficiency improvement over time.

    Parameters
    ----------
    env:
        The ``SequenceEnvironment`` instance.
    agent:
        The ``MuscleMemoryAgent`` to train.
    """

    def __init__(self, env: SequenceEnvironment, agent: MuscleMemoryAgent) -> None:
        self.env = env
        self.agent = agent
        self.stats = EpisodeStats()

    def run_episode(self, explore: bool = True) -> Dict[str, Any]:
        """
        Execute one complete episode and return its metrics.

        Parameters
        ----------
        explore:
            When ``True`` the agent uses epsilon-greedy exploration and the
            policy / muscle memory are updated after each step.

        Returns
        -------
        dict with keys: reward, nn_calls, cache_hits, success, duration
        """
        state = self.env.reset()
        total_reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        prev_nn = self.agent.nn_calls
        prev_cache = self.agent.cache_hits
        t0 = time.perf_counter()

        while not done:
            context = self.env.context_key()
            action, _from_cache = self.agent.act(state, context, explore=explore)
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward

            if explore:
                self.agent.learn(state, action, reward, context, info["correct"])

            state = next_state

        duration = time.perf_counter() - t0
        ep_nn_calls = self.agent.nn_calls - prev_nn
        ep_cache_hits = self.agent.cache_hits - prev_cache

        # Episode is a success when all steps were correct
        success = bool(
            info.get("step", 0) == info.get("seq_len", -1) and total_reward > 0
        )

        self.stats.record(total_reward, ep_nn_calls, ep_cache_hits, success, duration)
        return {
            "reward": total_reward,
            "nn_calls": ep_nn_calls,
            "cache_hits": ep_cache_hits,
            "success": success,
            "duration": duration,
        }

    def train(
        self,
        n_episodes: int,
        report_interval: int = 100,
    ) -> EpisodeStats:
        """
        Train the agent for ``n_episodes`` episodes.

        Parameters
        ----------
        n_episodes:
            Total number of training episodes.
        report_interval:
            Print a progress row every this many episodes.

        Returns
        -------
        EpisodeStats
            The accumulated statistics object.
        """
        header = (
            f"{'Episode':>8} | {'Reward':>8} | {'Success':>8} | "
            f"{'NN Calls':>9} | {'Cache Hits':>11} | {'Cache Rate':>11}"
        )
        sep = "-" * len(header)
        print(f"Training for {n_episodes} episodes…\n{header}\n{sep}")

        for ep in range(1, n_episodes + 1):
            self.run_episode(explore=True)

            if ep % report_interval == 0:
                ws = self.stats.window_stats(report_interval)
                print(
                    f"{ep:>8} | {ws['mean_reward']:>8.2f} | {ws['success_rate']:>8.1%} | "
                    f"{ws['mean_nn_calls']:>9.1f} | {ws['mean_cache_hits']:>11.1f} | "
                    f"{ws['cache_hit_rate']:>11.1%}"
                )

        print(f"\nTraining complete — {n_episodes} episodes.")
        return self.stats
