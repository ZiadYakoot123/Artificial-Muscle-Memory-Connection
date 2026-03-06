"""Tests for SequenceEnvironment."""

import numpy as np
import pytest

from muscle_memory.environment import SequenceEnvironment


SEQUENCES = [[0, 1, 2], [2, 1, 0, 1]]
N_ACTIONS = 3


@pytest.fixture
def env():
    return SequenceEnvironment(sequences=SEQUENCES, n_actions=N_ACTIONS)


class TestSequenceEnvironmentInit:
    def test_state_size(self, env):
        # state_size = n_actions + 2
        assert env.state_size == N_ACTIONS + 2

    def test_invalid_empty_sequences(self):
        with pytest.raises(ValueError):
            SequenceEnvironment(sequences=[], n_actions=3)

    def test_invalid_n_actions(self):
        with pytest.raises(ValueError):
            SequenceEnvironment(sequences=[[0, 1]], n_actions=0)

    def test_action_out_of_range(self):
        with pytest.raises(ValueError):
            SequenceEnvironment(sequences=[[0, 3]], n_actions=3)

    def test_empty_inner_sequence(self):
        with pytest.raises(ValueError):
            SequenceEnvironment(sequences=[[]], n_actions=3)


class TestReset:
    def test_returns_correct_shape(self, env):
        obs = env.reset()
        assert obs.shape == (env.state_size,)

    def test_specific_seq_idx(self, env):
        env.reset(seq_idx=0)
        assert env.current_sequence == SEQUENCES[0]

    def test_seq_idx_wraps_around(self, env):
        env.reset(seq_idx=len(SEQUENCES))
        assert env.current_sequence == SEQUENCES[0]

    def test_step_reset_to_zero(self, env):
        env.reset()
        assert env.current_step == 0


class TestStep:
    def test_correct_action_increments_step(self, env):
        env.reset(seq_idx=0)
        obs, reward, done, info = env.step(SEQUENCES[0][0])
        assert env.current_step == 1
        assert reward > 0
        assert info["correct"] is True

    def test_wrong_action_ends_episode(self, env):
        env.reset(seq_idx=0)
        wrong = (SEQUENCES[0][0] + 1) % N_ACTIONS
        _, reward, done, info = env.step(wrong)
        assert done is True
        assert reward < 0
        assert info["correct"] is False

    def test_completing_sequence_gives_bonus(self, env):
        env.reset(seq_idx=0)
        total_reward = 0.0
        done = False
        for action in SEQUENCES[0]:
            _, reward, done, _ = env.step(action)
            total_reward += reward
        assert done is True
        assert total_reward > len(SEQUENCES[0])  # bonus received

    def test_step_without_reset_raises(self):
        env = SequenceEnvironment(sequences=SEQUENCES, n_actions=N_ACTIONS)
        with pytest.raises(RuntimeError):
            env.step(0)


class TestContextKey:
    def test_key_type_is_tuple(self, env):
        env.reset()
        key = env.context_key()
        assert isinstance(key, tuple)

    def test_key_changes_after_step(self, env):
        env.reset(seq_idx=0)
        k0 = env.context_key()
        env.step(SEQUENCES[0][0])
        k1 = env.context_key()
        assert k0 != k1
