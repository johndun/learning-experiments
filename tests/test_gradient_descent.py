#!/usr/bin/env python3
"""
Tests for gradient descent implementation.

This module tests the manual gradient descent algorithm implementation,
including convergence properties, gradient computation accuracy, and edge cases.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add the scripts directory to the path so we can import the gradient descent functions
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from gradient_descent_demo import (
    analytical_gradient,
    gradient_descent,
    target_function,
)


class TestTargetFunction:
    """Test cases for the target function and its gradient."""

    def test_function_values(self):
        """Test target function evaluates correctly at known points."""
        # Test points and expected values
        test_cases = [
            (0.0, 10.0),  # f(0) = (0-3)^2 + 1 = 9 + 1 = 10
            (1.0, 5.0),  # f(1) = (1-3)^2 + 1 = 4 + 1 = 5
            (3.0, 1.0),  # f(3) = (3-3)^2 + 1 = 0 + 1 = 1 (minimum)
            (5.0, 5.0),  # f(5) = (5-3)^2 + 1 = 4 + 1 = 5
        ]

        for x_val, expected in test_cases:
            x = torch.tensor(x_val)
            result = target_function(x)
            assert torch.allclose(result, torch.tensor(expected)), (
                f"f({x_val}) should be {expected}, got {result.item()}"
            )

    def test_analytical_gradient(self):
        """Test analytical gradient computation."""
        # Test points and expected gradients
        test_cases = [
            (0.0, -6.0),  # df/dx(0) = 2(0-3) = -6
            (1.0, -4.0),  # df/dx(1) = 2(1-3) = -4
            (3.0, 0.0),  # df/dx(3) = 2(3-3) = 0 (critical point)
            (5.0, 4.0),  # df/dx(5) = 2(5-3) = 4
        ]

        for x_val, expected in test_cases:
            x = torch.tensor(x_val)
            result = analytical_gradient(x)
            assert torch.allclose(result, torch.tensor(expected)), (
                f"df/dx({x_val}) should be {expected}, got {result.item()}"
            )

    def test_gradient_vs_autodiff(self):
        """Test that analytical gradient matches automatic differentiation."""
        test_points = torch.tensor([0.0, 1.0, 2.5, 3.0, 4.5, 5.0])

        for x_val in test_points:
            # Analytical gradient
            analytical_grad = analytical_gradient(x_val)

            # Automatic differentiation gradient
            x_tensor = torch.tensor(x_val.item(), requires_grad=True)
            f_tensor = target_function(x_tensor)
            f_tensor.backward()
            auto_grad = x_tensor.grad

            assert torch.allclose(analytical_grad, auto_grad, atol=1e-6), (
                f"Analytical and auto grad should match at x={x_val.item()}: "
                f"analytical={analytical_grad.item()}, auto={auto_grad.item()}"
            )


class TestGradientDescent:
    """Test cases for the gradient descent algorithm."""

    def test_convergence_standard_case(self):
        """Test convergence from a standard starting point."""
        param_hist, func_hist = gradient_descent(
            initial_x=0.0, learning_rate=0.1, num_iterations=100, tolerance=1e-6
        )

        # Check that we have history
        assert len(param_hist) > 0, "Should have parameter history"
        assert len(func_hist) > 0, "Should have function value history"
        assert len(param_hist) == len(func_hist), "Histories should have same length"

        # Check convergence to minimum
        final_x = param_hist[-1]
        final_f = func_hist[-1]

        assert abs(final_x - 3.0) < 0.01, f"Should converge to x=3, got x={final_x}"
        assert abs(final_f - 1.0) < 0.01, f"Should converge to f=1, got f={final_f}"

    def test_convergence_different_starting_points(self):
        """Test convergence from different starting points."""
        starting_points = [-5.0, -1.0, 0.5, 7.0, 10.0]

        for start_x in starting_points:
            param_hist, func_hist = gradient_descent(
                initial_x=start_x, learning_rate=0.1, num_iterations=200, tolerance=1e-5
            )

            final_x = param_hist[-1]
            assert abs(final_x - 3.0) < 0.1, (
                f"Should converge to x≈3 from start={start_x}, got x={final_x}"
            )

    def test_learning_rate_effects(self):
        """Test the effect of different learning rates."""
        learning_rates = [0.01, 0.1, 0.3]
        convergence_errors = []

        for lr in learning_rates:
            param_hist, func_hist = gradient_descent(
                initial_x=0.0, learning_rate=lr, num_iterations=100, tolerance=1e-6
            )

            final_x = param_hist[-1]
            error = abs(final_x - 3.0)
            convergence_errors.append(error)

        # All should converge reasonably well
        for i, error in enumerate(convergence_errors):
            assert error < 0.5, f"Learning rate {learning_rates[i]} should converge reasonably"

    def test_monotonic_decrease(self):
        """Test that function values generally decrease during optimization."""
        param_hist, func_hist = gradient_descent(
            initial_x=0.0, learning_rate=0.1, num_iterations=50, tolerance=1e-6
        )

        # Function values should generally decrease (allowing for some small fluctuations)
        initial_f = func_hist[0]
        final_f = func_hist[-1]

        assert final_f < initial_f, "Function value should decrease overall"

        # Check that most consecutive pairs show decrease
        decreases = sum(1 for i in range(1, len(func_hist)) if func_hist[i] <= func_hist[i - 1])
        total_steps = len(func_hist) - 1

        # At least 80% of steps should show decrease or no increase
        assert decreases / total_steps >= 0.8, (
            f"Most steps should decrease function value: {decreases}/{total_steps}"
        )

    def test_parameter_tracking(self):
        """Test that parameter and function histories are correctly tracked."""
        param_hist, func_hist = gradient_descent(
            initial_x=1.0,
            learning_rate=0.1,
            num_iterations=10,
            tolerance=1e-10,  # Don't stop early
        )

        # Should have exactly 10 iterations recorded
        assert len(param_hist) == 10, f"Should have 10 parameter values, got {len(param_hist)}"
        assert len(func_hist) == 10, f"Should have 10 function values, got {len(func_hist)}"

        # First values should match initial conditions
        assert abs(param_hist[0] - 1.0) < 1e-6, "First parameter should be initial value"
        expected_first_f = target_function(torch.tensor(1.0)).item()
        assert abs(func_hist[0] - expected_first_f) < 1e-6, "First function value should match"

        # Function values should correspond to parameter values
        for i, (x_val, f_val) in enumerate(zip(param_hist, func_hist, strict=False)):
            expected_f = target_function(torch.tensor(x_val)).item()
            assert abs(f_val - expected_f) < 1e-5, (
                f"Function value at step {i} doesn't match: expected {expected_f}, got {f_val}"
            )

    def test_early_stopping(self):
        """Test that algorithm stops early when tolerance is reached."""
        # Use a generous tolerance to trigger early stopping
        param_hist, func_hist = gradient_descent(
            initial_x=2.9,  # Start close to minimum
            learning_rate=0.1,
            num_iterations=100,
            tolerance=1e-3,  # Generous tolerance
        )

        # Should stop well before 100 iterations
        assert len(param_hist) < 50, (
            f"Should stop early with generous tolerance, but took {len(param_hist)} iterations"
        )

    def test_edge_case_zero_learning_rate(self):
        """Test behavior with zero learning rate."""
        param_hist, func_hist = gradient_descent(
            initial_x=1.0, learning_rate=0.0, num_iterations=10, tolerance=1e-6
        )

        # Parameter should not change
        for x_val in param_hist:
            assert abs(x_val - 1.0) < 1e-6, "Parameter should not change with zero learning rate"

    def test_edge_case_at_minimum(self):
        """Test behavior when starting exactly at the minimum."""
        param_hist, func_hist = gradient_descent(
            initial_x=3.0,  # Exact minimum
            learning_rate=0.1,
            num_iterations=10,
            tolerance=1e-6,
        )

        # Should stop immediately or stay at minimum
        assert len(param_hist) <= 2, "Should stop quickly when starting at minimum"
        for x_val in param_hist:
            assert abs(x_val - 3.0) < 1e-3, "Should stay near minimum"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
