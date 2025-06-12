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
from pathlib import Path


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


def plot_function(
    x_range: Tuple[float, float] = (-2, 8),
    num_points: int = 1000,
    output_file: str = "gradient_descent_function.png"
) -> None:
    """
    Plot the target function over a specified range and save to file.

    Args:
        x_range: Tuple of (min_x, max_x) for plotting range
        num_points: Number of points to use for smooth curve
        output_file: Filename to save the plot to
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
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_file}")


def plot_function_with_optimization_path(
    param_history: List[float],
    function_history: List[float],
    x_range: Tuple[float, float] = (-2, 8),
    num_points: int = 1000,
    output_file: str = "gradient_descent_with_path.png"
) -> None:
    """
    Plot the target function with optimization path overlay.

    Args:
        param_history: List of parameter values during optimization
        function_history: List of function values during optimization
        x_range: Tuple of (min_x, max_x) for plotting range
        num_points: Number of points to use for smooth curve
        output_file: Filename to save the plot to
    """
    # Create function curve
    x_vals = torch.linspace(x_range[0], x_range[1], num_points)
    y_vals = target_function(x_vals)

    plt.figure(figsize=(12, 8))

    # Plot function curve
    plt.plot(x_vals.numpy(), y_vals.numpy(), 'b-', linewidth=2, label='f(x) = (x-3)² + 1')

    # Plot optimization path
    plt.plot(param_history, function_history, 'ro-', markersize=4, linewidth=1.5,
             alpha=0.8, label='Optimization Path')

    # Highlight start and end points
    plt.plot(param_history[0], function_history[0], 'go', markersize=10,
             label=f'Start: x={param_history[0]:.2f}')
    plt.plot(param_history[-1], function_history[-1], 'r*', markersize=12,
             label=f'End: x={param_history[-1]:.2f}')

    # Add step numbers for first few and last few steps
    step_labels = [0, 1, 2] + list(range(max(0, len(param_history)-3), len(param_history)))
    step_labels = sorted(set(step_labels))  # Remove duplicates and sort

    for i in step_labels:
        if i < len(param_history):
            plt.annotate(f'{i}',
                        (param_history[i], function_history[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

    # Add reference lines for true minimum
    plt.axvline(x=3, color='gray', linestyle='--', alpha=0.5, label='True minimum x=3')
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='True minimum f(x)=1')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent Optimization Path\nf(x) = (x - 3)² + 1')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimization path plot saved to: {output_file}")


def gradient_descent(
    initial_x: float,
    learning_rate: float = 0.1,
    num_iterations: int = 100,
    tolerance: float = 1e-6
) -> Tuple[List[float], List[float]]:
    """
    Manual gradient descent algorithm implementation using PyTorch's automatic differentiation.

    Args:
        initial_x: Starting point for optimization
        learning_rate: Step size for gradient descent
        num_iterations: Maximum number of iterations
        tolerance: Convergence tolerance (stop if gradient magnitude < tolerance)

    Returns:
        Tuple of (parameter_history, function_value_history)
    """
    # Initialize parameter as tensor with gradient tracking
    x = torch.tensor(initial_x, requires_grad=True, dtype=torch.float32)

    # Track optimization history
    param_history = []
    function_history = []

    print(f"Starting gradient descent from x = {initial_x:.4f}")
    print(f"Learning rate = {learning_rate}, Max iterations = {num_iterations}")
    print("-" * 60)

    for iteration in range(num_iterations):
        # Zero gradients from previous iteration
        if x.grad is not None:
            x.grad.zero_()

        # Compute function value
        f_val = target_function(x)

        # Compute gradient using automatic differentiation
        f_val.backward()

        # Store current state
        param_history.append(x.item())
        function_history.append(f_val.item())

        # Print progress every 10 iterations or at convergence
        if iteration % 10 == 0 or x.grad.abs() < tolerance:
            print(f"Iter {iteration:3d}: x = {x.item():8.5f}, "
                  f"f(x) = {f_val.item():8.5f}, "
                  f"grad = {x.grad.item():8.5f}")

        # Check convergence
        if x.grad.abs() < tolerance:
            print(f"Convergence achieved at iteration {iteration}!")
            break

        # Manual gradient descent update: x = x - learning_rate * gradient
        with torch.no_grad():
            x -= learning_rate * x.grad

        # Re-enable gradient tracking for next iteration
        x.requires_grad_(True)

    # Final state
    print(f"Final: x = {x.item():.5f}, f(x) = {target_function(x).item():.5f}")
    print(f"Theoretical minimum: x = 3.0, f(x) = 1.0")
    print(f"Error: |x - 3| = {abs(x.item() - 3.0):.6f}")

    return param_history, function_history


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


def plot_convergence_evolution(
    param_histories: List[List[float]],
    function_histories: List[List[float]],
    labels: List[str],
    output_file: str = "gradient_descent_convergence.png"
) -> None:
    """
    Plot parameter and loss evolution during gradient descent optimization.

    Args:
        param_histories: List of parameter histories for different test cases
        function_histories: List of function value histories for different test cases
        labels: List of labels for each test case
        output_file: Filename to save the plot to
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    colors = ['blue', 'red', 'green', 'purple', 'orange']

    # Plot parameter evolution
    for i, (param_hist, label) in enumerate(zip(param_histories, labels)):
        iterations = range(len(param_hist))
        color = colors[i % len(colors)]
        ax1.plot(iterations, param_hist, 'o-', color=color, markersize=3,
                linewidth=2, alpha=0.8, label=label)

    # Add horizontal line for true minimum
    ax1.axhline(y=3.0, color='gray', linestyle='--', alpha=0.7, label='True minimum (x=3)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Parameter Value (x)')
    ax1.set_title('Parameter Evolution During Gradient Descent')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot function value evolution (loss)
    for i, (func_hist, label) in enumerate(zip(function_histories, labels)):
        iterations = range(len(func_hist))
        color = colors[i % len(colors)]
        ax2.semilogy(iterations, func_hist, 'o-', color=color, markersize=3,
                    linewidth=2, alpha=0.8, label=label)

    # Add horizontal line for true minimum
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='True minimum (f=1)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Function Value (f(x)) - Log Scale')
    ax2.set_title('Loss Evolution During Gradient Descent')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Convergence evolution plot saved to: {output_file}")


def test_gradient_descent():
    """Test gradient descent convergence for the quadratic function."""
    print("\nTesting gradient descent convergence:")
    print("=" * 50)

    # Test cases with different starting points and learning rates
    test_cases = [
        {"initial_x": 0.0, "learning_rate": 0.1, "name": "Standard case (x=0, lr=0.1)"},
        {"initial_x": -2.0, "learning_rate": 0.05, "name": "Left start (x=-2, lr=0.05)"},
        {"initial_x": 8.0, "learning_rate": 0.2, "name": "Right start (x=8, lr=0.2)"},
    ]

    # Store histories for convergence plotting
    param_histories = []
    function_histories = []
    labels = []

    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {case['name']}")
        print("-" * 50)

        param_hist, func_hist = gradient_descent(
            initial_x=case["initial_x"],
            learning_rate=case["learning_rate"],
            num_iterations=50
        )

        # Store for convergence plotting
        param_histories.append(param_hist)
        function_histories.append(func_hist)
        labels.append(f"Start x={case['initial_x']}, lr={case['learning_rate']}")

        # Verify convergence
        final_x = param_hist[-1]
        final_f = func_hist[-1]
        error = abs(final_x - 3.0)

        print(f"Convergence test: Error = {error:.6f} (should be < 0.01)")
        print(f"Success: {error < 0.01}")

        if i < len(test_cases) - 1:
            print("\n" + "="*50)

    return param_histories, function_histories, labels


def main():
    """Main function to run the gradient descent demonstration."""
    print("Torch-Based Gradient Descent Demo")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print("Script initialized successfully!")
    print()
    print("Target Function:")
    print("f(x) = (x - 3)² + 1")
    print("Minimum at x = 3, f(3) = 1")
    print("Gradient: df/dx = 2(x - 3)")

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Test function and gradient computation
    test_function_and_gradient()

    # Test gradient descent algorithm and get histories for convergence plotting
    param_histories, function_histories, labels = test_gradient_descent()

    # Generate convergence evolution plots
    print("\nGenerating convergence evolution plots...")
    convergence_output_file = output_dir / "gradient_descent_convergence.png"
    plot_convergence_evolution(
        param_histories, function_histories, labels,
        output_file=str(convergence_output_file)
    )

    # Plot the target function and save to file
    print("\nPlotting target function...")
    output_file = output_dir / "gradient_descent_function.png"
    plot_function(output_file=str(output_file))

    # Demonstrate optimization path visualization
    print("\nGenerating optimization path visualization...")
    param_hist, func_hist = gradient_descent(
        initial_x=0.0,
        learning_rate=0.1,
        num_iterations=30
    )

    # Plot function with optimization path overlay
    path_output_file = output_dir / "gradient_descent_with_path.png"
    plot_function_with_optimization_path(
        param_hist, func_hist,
        output_file=str(path_output_file)
    )


if __name__ == "__main__":
    main()
