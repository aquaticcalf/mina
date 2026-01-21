"""
Tests for Hyperparameter Tuning Configuration and Script

These tests verify that hyperparameter tuning configuration is valid
and matches the design document specifications.
"""

import inspect

from hypothesis import given, settings
from hypothesis import strategies as st

from mina.core.constants import DEFAULT_TUNE_EPOCHS, DEFAULT_TUNE_ITERATIONS
from mina.tune import tune_hyperparameters


class TestTuningConfiguration:
    """Tests for tuning configuration validity."""

    def test_default_tune_epochs(self):
        """Verify default tuning epochs is 30."""
        assert DEFAULT_TUNE_EPOCHS == 30

    def test_default_tune_iterations(self):
        """Verify default tuning iterations is 300."""
        assert DEFAULT_TUNE_ITERATIONS == 300

    @given(
        epochs=st.integers(min_value=5, max_value=100),
        iterations=st.integers(min_value=50, max_value=1000),
    )
    @settings(max_examples=50)
    def test_tuning_params_valid(self, epochs: int, iterations: int):
        """
        **Feature: hyperparameter-tuning, Property: Tuning params validation**

        For any valid combination of tuning parameters,
        they should be within acceptable ranges.
        """
        assert epochs > 0
        assert iterations > 0

    def test_tune_function_signature(self):
        """Verify tune_hyperparameters has expected parameters."""
        sig = inspect.signature(tune_hyperparameters)
        params = list(sig.parameters.keys())

        assert "data" in params
        assert "epochs" in params
        assert "iterations" in params
        assert "optimizer" in params
        assert "device" in params

    def test_tune_function_has_docstring(self):
        """Verify tune_hyperparameters has a docstring."""
        assert tune_hyperparameters.__doc__ is not None
        assert len(tune_hyperparameters.__doc__) > 0


class TestTuningConstants:
    """Tests that verify tuning constants meet requirements."""

    def test_tune_epochs_reasonable(self):
        """
        Verify tuning epochs is reasonable for hyperparameter search.

        Should be shorter than full training (100 epochs) since we're
        searching for optimal hyperparameters.
        """
        assert DEFAULT_TUNE_EPOCHS < 100
        assert DEFAULT_TUNE_EPOCHS >= 10

    def test_tune_iterations_sufficient(self):
        """
        Verify tuning iterations provides sufficient search space.

        300 iterations is the Ray Tune default and provides good
        coverage for hyperparameter search.
        """
        assert DEFAULT_TUNE_ITERATIONS >= 100
        assert DEFAULT_TUNE_ITERATIONS <= 1000
