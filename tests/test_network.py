"""Tests for the MuscleMemoryNetwork class."""

import numpy as np
import pytest

from muscle_memory.network import MuscleMemoryNetwork
from muscle_memory.pattern import MotorPattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simple_pattern(n_steps: int = 4, n_units: int = 5, seed: int = 0) -> MotorPattern:
    return MotorPattern.random(
        n_steps=n_steps, n_units=n_units, sparsity=0.3, rng=np.random.default_rng(seed)
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestNetworkConstruction:
    def test_defaults(self):
        net = MuscleMemoryNetwork(n_units=4)
        assert net.n_units == 4
        assert net.learning_rate == 0.1
        assert net.decay_rate == 0.01

    def test_invalid_n_units(self):
        with pytest.raises(ValueError):
            MuscleMemoryNetwork(n_units=0)

    def test_invalid_learning_rate_zero(self):
        with pytest.raises(ValueError):
            MuscleMemoryNetwork(n_units=4, learning_rate=0.0)

    def test_invalid_learning_rate_above_one(self):
        with pytest.raises(ValueError):
            MuscleMemoryNetwork(n_units=4, learning_rate=1.1)

    def test_invalid_decay_rate_negative(self):
        with pytest.raises(ValueError):
            MuscleMemoryNetwork(n_units=4, decay_rate=-0.1)

    def test_invalid_decay_rate_one(self):
        with pytest.raises(ValueError):
            MuscleMemoryNetwork(n_units=4, decay_rate=1.0)

    def test_no_self_connections(self):
        net = MuscleMemoryNetwork(n_units=6)
        assert np.all(np.diag(net.weights) == 0.0)

    def test_initial_weights_small(self):
        net = MuscleMemoryNetwork(n_units=10, rng=np.random.default_rng(0))
        assert net.weights.max() < 0.1


# ---------------------------------------------------------------------------
# Practice
# ---------------------------------------------------------------------------

class TestPractice:
    def test_weights_increase_after_practice(self):
        net = MuscleMemoryNetwork(n_units=5, learning_rate=0.2,
                                   rng=np.random.default_rng(7))
        p = simple_pattern(n_units=5, seed=7)
        before = net.weights.sum()
        net.practice(p)
        assert net.weights.sum() > before

    def test_weights_clipped_to_one(self):
        net = MuscleMemoryNetwork(n_units=4, learning_rate=1.0,
                                   rng=np.random.default_rng(0))
        p = MotorPattern(np.ones((3, 4)))
        net.practice(p, n_trials=100)
        assert net.weights.max() <= 1.0

    def test_no_self_connections_after_practice(self):
        net = MuscleMemoryNetwork(n_units=5)
        p = simple_pattern(n_units=5)
        net.practice(p, n_trials=5)
        assert np.all(np.diag(net.weights) == 0.0)

    def test_practice_count_tracked(self):
        net = MuscleMemoryNetwork(n_units=5)
        p = simple_pattern(n_units=5)
        p2 = MotorPattern(np.ones((2, 5)) * 0.5, name="crouch")
        net.practice(p, n_trials=3)
        net.practice(p2, n_trials=1)
        net.practice(p2, n_trials=2)
        assert net.practice_counts["crouch"] == 3

    def test_mismatched_units_raises(self):
        net = MuscleMemoryNetwork(n_units=4)
        p = simple_pattern(n_units=6)
        with pytest.raises(ValueError, match="units"):
            net.practice(p)

    def test_more_trials_means_stronger_weights(self):
        rng = np.random.default_rng(42)
        p = simple_pattern(n_units=5, seed=42)

        net1 = MuscleMemoryNetwork(n_units=5, learning_rate=0.1, rng=np.random.default_rng(1))
        net2 = MuscleMemoryNetwork(n_units=5, learning_rate=0.1, rng=np.random.default_rng(1))
        net1.practice(p, n_trials=1)
        net2.practice(p, n_trials=10)
        assert net2.weights.sum() > net1.weights.sum()


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------

class TestRecall:
    def test_recall_returns_motor_pattern(self):
        net = MuscleMemoryNetwork(n_units=5)
        p = simple_pattern(n_units=5)
        net.practice(p, n_trials=10)
        cue = p.activations[0]
        recalled = net.recall(cue, n_steps=3)
        assert isinstance(recalled, MotorPattern)
        assert recalled.n_steps == 4  # cue + 3 generated steps
        assert recalled.n_units == 5

    def test_recall_output_in_range(self):
        net = MuscleMemoryNetwork(n_units=6)
        p = simple_pattern(n_units=6)
        net.practice(p, n_trials=5)
        cue = p.activations[0]
        recalled = net.recall(cue, n_steps=5)
        assert recalled.activations.min() >= 0.0
        assert recalled.activations.max() <= 1.0

    def test_recall_first_row_equals_cue(self):
        net = MuscleMemoryNetwork(n_units=5)
        cue = np.array([0.1, 0.9, 0.5, 0.0, 0.3])
        recalled = net.recall(cue, n_steps=2)
        np.testing.assert_array_equal(recalled.activations[0], cue)

    def test_recall_wrong_cue_shape_raises(self):
        net = MuscleMemoryNetwork(n_units=5)
        with pytest.raises(ValueError, match="shape"):
            net.recall(np.array([0.1, 0.2, 0.3]), n_steps=2)

    def test_recall_improves_with_practice(self):
        """After more practice the recalled pattern's active units should be
        more strongly activated (larger mean activation), because Hebbian
        learning pushes the sigmoid output higher at co-activated units."""
        rng = np.random.default_rng(0)
        p = MotorPattern.random(n_steps=5, n_units=8, sparsity=0.4, rng=rng, name="move")
        cue = p.activations[0]

        net_low = MuscleMemoryNetwork(n_units=8, learning_rate=0.2, rng=np.random.default_rng(0))
        net_high = MuscleMemoryNetwork(n_units=8, learning_rate=0.2, rng=np.random.default_rng(0))

        net_low.practice(p, n_trials=1)
        net_high.practice(p, n_trials=50)

        # More practice → stronger weights → sigmoid pushes activations higher
        mean_high = net_high.recall(cue, n_steps=4).activations[1:].mean()
        mean_low = net_low.recall(cue, n_steps=4).activations[1:].mean()
        assert mean_high >= mean_low


# ---------------------------------------------------------------------------
# Forget
# ---------------------------------------------------------------------------

class TestForget:
    def test_weights_decrease_after_forget(self):
        net = MuscleMemoryNetwork(n_units=5, decay_rate=0.1)
        p = simple_pattern(n_units=5)
        net.practice(p, n_trials=10)
        before = net.weights.sum()
        net.forget(n_steps=1)
        assert net.weights.sum() < before

    def test_forget_multiple_steps(self):
        net = MuscleMemoryNetwork(n_units=5, decay_rate=0.1)
        p = simple_pattern(n_units=5)
        net.practice(p, n_trials=10)
        before = net.weights.sum()
        net.forget(n_steps=10)
        assert net.weights.sum() < before

    def test_forget_invalid_n_steps(self):
        net = MuscleMemoryNetwork(n_units=5)
        with pytest.raises(ValueError):
            net.forget(n_steps=0)

    def test_forget_zero_decay_rate_no_change(self):
        net = MuscleMemoryNetwork(n_units=5, decay_rate=0.0)
        p = simple_pattern(n_units=5)
        net.practice(p, n_trials=5)
        w_before = net.weights.copy()
        net.forget(n_steps=100)
        np.testing.assert_array_equal(net.weights, w_before)


# ---------------------------------------------------------------------------
# Connection strength / strongest connections
# ---------------------------------------------------------------------------

class TestConnectionStrength:
    def test_connection_strength_returns_float(self):
        net = MuscleMemoryNetwork(n_units=4)
        val = net.connection_strength(0, 1)
        assert isinstance(val, float)

    def test_strongest_connections_length(self):
        net = MuscleMemoryNetwork(n_units=6)
        p = simple_pattern(n_units=6)
        net.practice(p, n_trials=5)
        top = net.strongest_connections(top_k=3)
        assert len(top) == 3

    def test_strongest_connections_sorted(self):
        net = MuscleMemoryNetwork(n_units=6)
        p = simple_pattern(n_units=6)
        net.practice(p, n_trials=5)
        top = net.strongest_connections(top_k=5)
        weights = [w for _, _, w in top]
        assert weights == sorted(weights, reverse=True)
