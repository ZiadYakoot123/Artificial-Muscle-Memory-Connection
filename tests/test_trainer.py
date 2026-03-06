"""Tests for Trainer and EpisodeStats."""

import numpy as np
import pytest

from muscle_memory.agent import MuscleMemoryAgent
from muscle_memory.environment import SequenceEnvironment
from muscle_memory.trainer import EpisodeStats, Trainer


SEQUENCES = [[0, 1, 2], [1, 0]]
N_ACTIONS = 3


@pytest.fixture
def env():
    return SequenceEnvironment(sequences=SEQUENCES, n_actions=N_ACTIONS)


@pytest.fixture
def agent(env):
    return MuscleMemoryAgent(
        state_size=env.state_size,
        n_actions=N_ACTIONS,
        hidden_size=16,
        lr=0.01,
        familiarity_threshold=5,
        epsilon=0.0,
    )


@pytest.fixture
def trainer(env, agent):
    return Trainer(env=env, agent=agent)


class TestEpisodeStats:
    def test_len_zero_initially(self):
        stats = EpisodeStats()
        assert len(stats) == 0

    def test_record_increments_length(self):
        stats = EpisodeStats()
        stats.record(5.0, 3, 1, True, 0.01)
        assert len(stats) == 1

    def test_window_stats_empty(self):
        stats = EpisodeStats()
        assert stats.window_stats() == {}

    def test_window_stats_values(self):
        stats = EpisodeStats()
        stats.record(10.0, 4, 0, True, 0.01)
        stats.record(-1.0, 3, 0, False, 0.01)
        ws = stats.window_stats(2)
        assert abs(ws["mean_reward"] - 4.5) < 1e-9
        assert abs(ws["success_rate"] - 0.5) < 1e-9

    def test_window_stats_range_empty(self):
        stats = EpisodeStats()
        assert stats.window_stats_range(0, 100) == {}

    def test_window_stats_range(self):
        stats = EpisodeStats()
        for _ in range(10):
            stats.record(1.0, 2, 0, True, 0.0)
        ws = stats.window_stats_range(0, 5)
        assert ws["mean_reward"] == pytest.approx(1.0)
        assert ws["success_rate"] == pytest.approx(1.0)

    def test_window_stats_range_clamped(self):
        stats = EpisodeStats()
        stats.record(2.0, 1, 0, True, 0.0)
        ws = stats.window_stats_range(0, 999)
        assert ws["mean_reward"] == pytest.approx(2.0)


class TestTrainer:
    def test_run_episode_returns_dict(self, trainer):
        result = trainer.run_episode()
        assert "reward" in result
        assert "nn_calls" in result
        assert "cache_hits" in result
        assert "success" in result
        assert "duration" in result

    def test_run_episode_updates_stats(self, trainer):
        trainer.run_episode()
        assert len(trainer.stats) == 1

    def test_run_episode_no_explore(self, trainer):
        """Non-exploration episode should not update memory."""
        ctx_before = trainer.agent.memory.n_tracked
        trainer.run_episode(explore=False)
        assert trainer.agent.memory.n_tracked == ctx_before

    def test_train_runs_n_episodes(self, trainer, capsys):
        trainer.train(n_episodes=10, report_interval=5)
        assert len(trainer.stats) == 10

    def test_train_prints_header(self, trainer, capsys):
        trainer.train(n_episodes=5, report_interval=5)
        out = capsys.readouterr().out
        assert "Episode" in out

    def test_efficiency_improves_over_training(self, env):
        """Cache hit rate should be higher in late training than early."""
        np.random.seed(0)
        agent = MuscleMemoryAgent(
            state_size=env.state_size,
            n_actions=N_ACTIONS,
            hidden_size=16,
            lr=0.01,
            familiarity_threshold=5,
            epsilon=0.0,
        )
        trainer = Trainer(env=env, agent=agent)
        trainer.train(n_episodes=200, report_interval=200)
        early = trainer.stats.window_stats_range(0, 20)
        late = trainer.stats.window_stats_range(180, 200)
        # After enough training the cache hit rate should be higher
        assert late["cache_hit_rate"] >= early["cache_hit_rate"]
