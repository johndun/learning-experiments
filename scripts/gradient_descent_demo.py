#!/usr/bin/env python3
"""
Torch-Based Gradient Descent Implementation Demo

This script demonstrates manual gradient descent implementation using PyTorch's
automatic differentiation capabilities. It includes:

1. A simple quadratic function optimization example
2. Manual gradient descent algorithm implementation
3. Visualization of the optimization process
4. Comparison with PyTorch's built-in optimizers

The purpose is to provide a clear understanding of how gradient descent works
under the hood, while leveraging PyTorch's tensor operations and automatic
differentiation for accurate gradient computation.

Author: Learning Experiments
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional


def target_function(x: torch.Tensor) -> torch.Tensor:
    """
    Target function to optimize: f(x) = (x - 3)^2 + 1

    This is a simple quadratic function with minimum at x = 3, f(3) = 1.

    Args:
        x: Input tensor (can be scalar or vector)

    Returns:
        Function value at x
    """
    return (x - 3) ** 2 + 1


def analytical_gradient(x: torch.Tensor) -> torch.Tensor:
    """
    Analytical gradient of target function: df/dx = 2(x - 3)

    Args:
        x: Input tensor

    Returns:
        Gradient value at x
    """
    return 2 * (x - 3)


def plot_function(x_range: Tuple[float, float] = (-2, 8), num_points: int = 1000) -> None:
    """
    Plot the target function over a specified range.

    Args:
        x_range: Tuple of (min_x, max_x) for plotting range
        num_points: Number of points to use for smooth curve
    """
    x_vals = torch.linspace(x_range[0], x_range[1], num_points)
    y_vals = target_function(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals.numpy(), y_vals.numpy(), 'b-', linewidth=2, label='f(x) = (x-3)² + 1')
    plt.axvline(x=3, color='r', linestyle='--', alpha=0.7, label='Minimum at x=3')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Minimum value f(3)=1')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Target Function: f(x) = (x - 3)² + 1')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def test_function_and_gradient():
    """Test function evaluation and gradient computation."""
    print("\nTesting function and gradient computation:")
    print("-" * 40)

    # Test points
    test_points = torch.tensor([0.0, 1.0, 3.0, 5.0])

    for x_val in test_points:
        # Function value
        f_val = target_function(x_val)

        # Analytical gradient
        analytical_grad = analytical_gradient(x_val)

        # Automatic differentiation gradient
        x_tensor = torch.tensor(x_val.item(), requires_grad=True)
        f_tensor = target_function(x_tensor)
        f_tensor.backward()
        auto_grad = x_tensor.grad

        print(f"x = {x_val.item():4.1f}: f(x) = {f_val.item():6.3f}, "
              f"analytical grad = {analytical_grad.item():6.3f}, "
              f"auto grad = {auto_grad.item():6.3f}, "
              f"match = {torch.allclose(analytical_grad, auto_grad)}")


def main():
    """Main function to run the gradient descent demonstration."""
    print("Torch-Based Gradient Descent Demo")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print("Script initialized successfully!")

    # Test function and gradient computation
    test_function_and_gradient()

    # Plot the target function
    print("\nPlotting target function...")
    plot_function()


if __name__ == "__main__":
    main()
