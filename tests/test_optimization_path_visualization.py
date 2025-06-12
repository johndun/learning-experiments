#!/usr/bin/env python3
"""
Tests for optimization path visualization functionality.

This test module verifies the correct implementation of:
1. Function curve plotting with optimization path overlay
2. Proper visualization formatting and output generation
3. Correct handling of optimization history data
"""

import os
import sys
import tempfile

import torch

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from gradient_descent_demo import (
    gradient_descent,
    plot_function_with_optimization_path,
    target_function,
)


def test_optimization_path_generation():
    """Test that gradient descent generates valid optimization history."""
    print("Testing optimization path generation...")

    # Run gradient descent
    param_hist, func_hist = gradient_descent(initial_x=0.0, learning_rate=0.1, num_iterations=10)

    # Verify history lists are populated
    assert len(param_hist) > 0, "Parameter history should not be empty"
    assert len(func_hist) > 0, "Function history should not be empty"
    assert len(param_hist) == len(func_hist), "History lists should have same length"

    # Verify optimization progresses toward minimum
    initial_error = abs(param_hist[0] - 3.0)
    final_error = abs(param_hist[-1] - 3.0)
    assert final_error < initial_error, "Should converge toward minimum at x=3"

    # Verify function values decrease
    assert func_hist[-1] < func_hist[0], "Function value should decrease during optimization"

    print(f"✓ Generated {len(param_hist)} optimization steps")
    print(f"✓ Initial error: {initial_error:.4f}, Final error: {final_error:.4f}")
    print(f"✓ Function value decreased from {func_hist[0]:.4f} to {func_hist[-1]:.4f}")


def test_visualization_plot_creation():
    """Test that optimization path visualization creates valid plot files."""
    print("\nTesting visualization plot creation...")

    # Generate test data
    param_hist, func_hist = gradient_descent(initial_x=-1.0, learning_rate=0.15, num_iterations=8)

    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        output_file = tmp_file.name

    try:
        # Test plot creation
        plot_function_with_optimization_path(param_hist, func_hist, output_file=output_file)

        # Verify file was created
        assert os.path.exists(output_file), "Plot file should be created"

        # Verify file has content (not empty)
        file_size = os.path.getsize(output_file)
        assert file_size > 1000, f"Plot file should have substantial content, got {file_size} bytes"

        print(f"✓ Plot file created successfully: {file_size} bytes")

    finally:
        # Clean up temporary file
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_visualization_with_different_starting_points():
    """Test visualization works with different optimization starting points."""
    print("\nTesting visualization with different starting points...")

    test_cases = [
        {"initial_x": -2.0, "learning_rate": 0.05, "name": "Left start"},
        {"initial_x": 5.0, "learning_rate": 0.1, "name": "Right start"},
        {"initial_x": 2.9, "learning_rate": 0.2, "name": "Near minimum"},
    ]

    for case in test_cases:
        param_hist, func_hist = gradient_descent(
            initial_x=case["initial_x"], learning_rate=case["learning_rate"], num_iterations=5
        )

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            output_file = tmp_file.name

        try:
            # Test plot creation
            plot_function_with_optimization_path(param_hist, func_hist, output_file=output_file)

            # Verify file was created
            assert os.path.exists(output_file), f"Plot should be created for {case['name']}"

            print(f"✓ {case['name']} visualization created successfully")

        finally:
            # Clean up
            if os.path.exists(output_file):
                os.unlink(output_file)


def test_optimization_path_mathematical_properties():
    """Test that optimization path has correct mathematical properties."""
    print("\nTesting optimization path mathematical properties...")

    # Generate optimization path
    param_hist, func_hist = gradient_descent(initial_x=1.0, learning_rate=0.1, num_iterations=15)

    # Verify all function values match expected computation
    for i, (x_val, f_val) in enumerate(zip(param_hist, func_hist, strict=False)):
        expected_f = target_function(torch.tensor(x_val)).item()
        error = abs(f_val - expected_f)
        assert error < 1e-5, f"Function value mismatch at step {i}: {f_val} vs {expected_f}"

    # Verify optimization generally moves toward minimum
    mid_point = len(param_hist) // 2
    early_error = abs(param_hist[mid_point] - 3.0)
    late_error = abs(param_hist[-1] - 3.0)
    assert late_error <= early_error, "Should not move away from minimum in later steps"

    print(f"✓ All {len(param_hist)} function values match expected computation")
    print(f"✓ Optimization converged: early error {early_error:.4f} → late error {late_error:.4f}")


def main():
    """Run all optimization path visualization tests."""
    print("Optimization Path Visualization Tests")
    print("=" * 40)

    # Run all test functions
    test_functions = [
        test_optimization_path_generation,
        test_visualization_plot_creation,
        test_visualization_with_different_starting_points,
        test_optimization_path_mathematical_properties,
    ]

    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
            return False

    print("\n" + "=" * 40)
    print("✓ All optimization path visualization tests passed!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
