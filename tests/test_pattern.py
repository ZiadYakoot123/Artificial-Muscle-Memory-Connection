"""Tests for the MotorPattern class."""

import numpy as np
import pytest

from muscle_memory.pattern import MotorPattern


class TestMotorPatternInit:
    def test_valid_construction(self):
        data = np.array([[0.1, 0.9], [0.5, 0.5], [0.0, 1.0]])
        p = MotorPattern(data, name="test")
        assert p.n_steps == 3
        assert p.n_units == 2
        assert p.name == "test"

    def test_activations_are_read_only_copy(self):
        data = np.array([[0.2, 0.8], [0.4, 0.6]])
        p = MotorPattern(data)
        # Modifying the original array should not affect the stored pattern
        data[0, 0] = 0.99
        assert p.activations[0, 0] != 0.99

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            MotorPattern(np.array([0.1, 0.2, 0.3]))

    def test_rejects_negative_values(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            MotorPattern(np.array([[-0.1, 0.5], [0.3, 0.7]]))

    def test_rejects_values_above_one(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            MotorPattern(np.array([[1.1, 0.5], [0.3, 0.7]]))

    def test_boundary_values_accepted(self):
        data = np.array([[0.0, 1.0], [1.0, 0.0]])
        p = MotorPattern(data)
        assert p.n_steps == 2

    def test_default_name_is_empty_string(self):
        p = MotorPattern(np.array([[0.5, 0.5]]))
        assert p.name == ""


class TestMotorPatternRandom:
    def test_shape(self):
        p = MotorPattern.random(n_steps=4, n_units=10, rng=np.random.default_rng(0))
        assert p.n_steps == 4
        assert p.n_units == 10

    def test_values_in_range(self):
        rng = np.random.default_rng(1)
        p = MotorPattern.random(n_steps=20, n_units=20, rng=rng)
        assert p.activations.min() >= 0.0
        assert p.activations.max() <= 1.0

    def test_sparsity_applied(self):
        # With sparsity=1.0 all activations should be 0
        rng = np.random.default_rng(2)
        p = MotorPattern.random(n_steps=5, n_units=5, sparsity=1.0, rng=rng)
        assert np.all(p.activations == 0.0)

    def test_reproducibility_with_same_seed(self):
        p1 = MotorPattern.random(4, 6, rng=np.random.default_rng(99))
        p2 = MotorPattern.random(4, 6, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(p1.activations, p2.activations)

    def test_name_passed_through(self):
        p = MotorPattern.random(2, 2, name="jump", rng=np.random.default_rng(0))
        assert p.name == "jump"
