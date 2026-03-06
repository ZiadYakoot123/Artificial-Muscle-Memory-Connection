"""Tests for MuscleMemory."""

import pytest

from muscle_memory.memory import MuscleMemory


class TestMuscleMemoryInit:
    def test_default_threshold(self):
        mm = MuscleMemory()
        assert mm.familiarity_threshold == 10

    def test_custom_threshold(self):
        mm = MuscleMemory(familiarity_threshold=3)
        assert mm.familiarity_threshold == 3

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            MuscleMemory(familiarity_threshold=0)


class TestUpdate:
    def test_unsuccessful_update_does_not_cache(self):
        mm = MuscleMemory(familiarity_threshold=2)
        ctx = (0, 0)
        mm.update(ctx, action=1, success=False)
        mm.update(ctx, action=1, success=False)
        assert mm.lookup(ctx) is None

    def test_successful_updates_below_threshold_not_cached(self):
        mm = MuscleMemory(familiarity_threshold=3)
        ctx = (0, 0)
        mm.update(ctx, action=1, success=True)
        mm.update(ctx, action=1, success=True)
        assert mm.lookup(ctx) is None

    def test_reaches_threshold_caches_action(self):
        mm = MuscleMemory(familiarity_threshold=3)
        ctx = (0, 0)
        for _ in range(3):
            mm.update(ctx, action=2, success=True)
        assert mm.lookup(ctx) == 2

    def test_n_cached_increments_on_threshold(self):
        mm = MuscleMemory(familiarity_threshold=2)
        ctx_a = (0, 0)
        ctx_b = (0, 1)
        for _ in range(2):
            mm.update(ctx_a, action=0, success=True)
        assert mm.n_cached == 1
        for _ in range(2):
            mm.update(ctx_b, action=1, success=True)
        assert mm.n_cached == 2


class TestLookup:
    def test_unknown_context_returns_none(self):
        mm = MuscleMemory(familiarity_threshold=5)
        assert mm.lookup((99, 99)) is None

    def test_cached_context_returns_action(self):
        mm = MuscleMemory(familiarity_threshold=1)
        mm.update((0, 0), action=1, success=True)
        assert mm.lookup((0, 0)) == 1


class TestStatistics:
    def test_familiarity_ratio_zero_when_empty(self):
        mm = MuscleMemory()
        assert mm.familiarity_ratio() == 0.0

    def test_familiarity_ratio_partial(self):
        mm = MuscleMemory(familiarity_threshold=1)
        mm.update((0, 0), action=0, success=True)   # cached
        mm.update((0, 1), action=1, success=False)  # not cached (tracked but count=0)
        # n_cached=1, n_tracked=1 (only ctx with success>0 is counted)
        # Actually both contexts are tracked, but only (0,0) is cached
        ratio = mm.familiarity_ratio()
        assert 0.0 < ratio <= 1.0

    def test_is_cached(self):
        mm = MuscleMemory(familiarity_threshold=1)
        ctx = (1, 2)
        assert not mm.is_cached(ctx)
        mm.update(ctx, action=0, success=True)
        assert mm.is_cached(ctx)

    def test_familiarity_count(self):
        mm = MuscleMemory(familiarity_threshold=5)
        ctx = (0, 0)
        for _ in range(3):
            mm.update(ctx, action=2, success=True)
        assert mm.familiarity_count(ctx, 2) == 3
        assert mm.familiarity_count(ctx, 0) == 0
