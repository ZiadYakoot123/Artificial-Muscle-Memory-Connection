"""Tests for NeuralPolicy and MuscleMemoryAgent."""

import numpy as np
import pytest

from muscle_memory.agent import MuscleMemoryAgent, NeuralPolicy


STATE_SIZE = 5
N_ACTIONS = 3
HIDDEN_SIZE = 8


@pytest.fixture
def policy():
    return NeuralPolicy(STATE_SIZE, HIDDEN_SIZE, N_ACTIONS, lr=0.01)


@pytest.fixture
def agent():
    return MuscleMemoryAgent(
        state_size=STATE_SIZE,
        n_actions=N_ACTIONS,
        hidden_size=HIDDEN_SIZE,
        lr=0.01,
        familiarity_threshold=3,
        epsilon=0.0,  # deterministic for tests
    )


class TestNeuralPolicy:
    def test_forward_output_shape(self, policy):
        state = np.zeros(STATE_SIZE)
        probs = policy.forward(state)
        assert probs.shape == (N_ACTIONS,)

    def test_forward_output_sums_to_one(self, policy):
        state = np.random.rand(STATE_SIZE)
        probs = policy.forward(state)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_forward_all_probabilities_positive(self, policy):
        state = np.random.rand(STATE_SIZE)
        probs = policy.forward(state)
        assert np.all(probs > 0)

    def test_update_changes_weights(self, policy):
        state = np.random.rand(STATE_SIZE)
        w1_before = policy.W1.copy()
        policy.update(state, action=0, advantage=1.0)
        assert not np.allclose(policy.W1, w1_before)

    def test_update_zero_advantage_no_change(self, policy):
        state = np.random.rand(STATE_SIZE)
        # First forward to cache activations
        w1_before = policy.W1.copy()
        policy.update(state, action=0, advantage=0.0)
        # With advantage=0 the gradient is zero → no weight change
        assert np.allclose(policy.W1, w1_before)


class TestMuscleMemoryAgent:
    def test_act_returns_valid_action(self, agent):
        state = np.zeros(STATE_SIZE)
        action, from_cache = agent.act(state, context=(0, 0), explore=False)
        assert 0 <= action < N_ACTIONS
        assert from_cache is False

    def test_act_uses_cache_when_available(self, agent):
        # Manually push an action into the muscle memory cache
        agent.memory._cache[(0, 0)] = 2
        state = np.zeros(STATE_SIZE)
        action, from_cache = agent.act(state, context=(0, 0), explore=False)
        assert action == 2
        assert from_cache is True

    def test_cache_hit_increments_counter(self, agent):
        agent.memory._cache[(1, 1)] = 0
        state = np.zeros(STATE_SIZE)
        before = agent.cache_hits
        agent.act(state, context=(1, 1))
        assert agent.cache_hits == before + 1

    def test_nn_call_increments_counter(self, agent):
        state = np.zeros(STATE_SIZE)
        before = agent.nn_calls
        agent.act(state, context=(99, 99), explore=False)
        assert agent.nn_calls == before + 1

    def test_cache_hit_rate_zero_initially(self, agent):
        assert agent.cache_hit_rate == 0.0

    def test_cache_hit_rate_after_calls(self, agent):
        # Force cache for context (0, 0) and make two calls
        agent.memory._cache[(0, 0)] = 0
        state = np.zeros(STATE_SIZE)
        agent.act(state, context=(0, 0))   # cache hit
        agent.act(state, context=(1, 1))   # nn call
        assert abs(agent.cache_hit_rate - 0.5) < 1e-9

    def test_learn_updates_memory(self, agent):
        state = np.zeros(STATE_SIZE)
        ctx = (0, 0)
        for _ in range(3):
            agent.learn(state, action=1, reward=1.0, context=ctx, success=True)
        assert agent.memory.is_cached(ctx)

    def test_efficiency_gain_equals_cache_hit_rate(self, agent):
        agent.memory._cache[(0, 0)] = 0
        state = np.zeros(STATE_SIZE)
        agent.act(state, (0, 0))
        assert agent.efficiency_gain == agent.cache_hit_rate
