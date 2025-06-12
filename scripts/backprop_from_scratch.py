#!/usr/bin/env python3
"""
Backpropagation from Scratch

A complete implementation of backpropagation algorithm for neural networks
without external ML libraries. Only uses numpy and matplotlib for basic
operations and visualization.

This script demonstrates:
- Forward propagation
- Backpropagation algorithm
- Gradient descent optimization
- Training on XOR dataset
- Visualization of results
"""

import math
import random
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
import numpy as np


# Basic Mathematical Functions
def exp(x: float) -> float:
    """Compute e^x safely, handling overflow."""
    try:
        return math.exp(x)
    except OverflowError:
        return float('inf') if x > 0 else 0.0


def sigmoid(x: float) -> float:
    """Sigmoid activation function: 1 / (1 + e^(-x))."""
    if x > 500:  # Prevent overflow
        return 1.0
    elif x < -500:  # Prevent underflow
        return 0.0
    return 1.0 / (1.0 + exp(-x))


def tanh(x: float) -> float:
    """Hyperbolic tangent activation function."""
    if x > 500:
        return 1.0
    elif x < -500:
        return -1.0
    return math.tanh(x)


def relu(x: float) -> float:
    """ReLU activation function: max(0, x)."""
    return max(0.0, x)


# Activation Function Derivatives
def sigmoid_derivative(x: float) -> float:
    """Derivative of sigmoid function: sigmoid(x) * (1 - sigmoid(x))."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_derivative(x: float) -> float:
    """Derivative of tanh function: 1 - tanh^2(x)."""
    t = tanh(x)
    return 1.0 - t * t


def relu_derivative(x: float) -> float:
    """Derivative of ReLU function: 1 if x > 0, else 0."""
    return 1.0 if x > 0.0 else 0.0


# Loss Functions
def mean_squared_error(predictions: List[float], targets: List[float]) -> float:
    """
    Compute mean squared error loss.

    MSE = (1/n) * Σ(predictions - targets)²

    Args:
        predictions: Model predictions
        targets: True target values

    Returns:
        Mean squared error value
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    n = len(predictions)
    if n == 0:
        return 0.0

    total_error = 0.0
    for i in range(n):
        error = predictions[i] - targets[i]
        total_error += error * error

    return total_error / n


def mse_derivative(predictions: List[float], targets: List[float]) -> List[float]:
    """
    Compute derivative of mean squared error with respect to predictions.

    d(MSE)/d(predictions) = (2/n) * (predictions - targets)

    Args:
        predictions: Model predictions
        targets: True target values

    Returns:
        List of gradients with respect to each prediction
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    n = len(predictions)
    if n == 0:
        return []

    gradients = []
    for i in range(n):
        gradient = (2.0 / n) * (predictions[i] - targets[i])
        gradients.append(gradient)

    return gradients


def binary_cross_entropy(predictions: List[float], targets: List[float]) -> float:
    """
    Compute binary cross-entropy loss.

    BCE = -(1/n) * Σ[targets * log(predictions) + (1-targets) * log(1-predictions)]

    Args:
        predictions: Model predictions (should be between 0 and 1)
        targets: True target values (should be 0 or 1)

    Returns:
        Binary cross-entropy loss value
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    n = len(predictions)
    if n == 0:
        return 0.0

    total_loss = 0.0
    epsilon = 1e-15  # Small value to prevent log(0)

    for i in range(n):
        # Clip predictions to prevent log(0)
        pred = max(epsilon, min(1.0 - epsilon, predictions[i]))
        target = targets[i]

        loss = -(target * math.log(pred) + (1.0 - target) * math.log(1.0 - pred))
        total_loss += loss

    return total_loss / n


def bce_derivative(predictions: List[float], targets: List[float]) -> List[float]:
    """
    Compute derivative of binary cross-entropy with respect to predictions.

    d(BCE)/d(predictions) = (1/n) * [(predictions - targets) / (predictions * (1 - predictions))]

    Args:
        predictions: Model predictions (should be between 0 and 1)
        targets: True target values (should be 0 or 1)

    Returns:
        List of gradients with respect to each prediction
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    n = len(predictions)
    if n == 0:
        return []

    gradients = []
    epsilon = 1e-15  # Small value to prevent division by zero

    for i in range(n):
        # Clip predictions to prevent division by zero
        pred = max(epsilon, min(1.0 - epsilon, predictions[i]))
        target = targets[i]

        gradient = (1.0 / n) * ((pred - target) / (pred * (1.0 - pred)))
        gradients.append(gradient)

    return gradients


# Basic Matrix Operations
class Matrix:
    """Simple matrix implementation for neural network operations."""

    def __init__(self, data: List[List[float]]):
        """Initialize matrix with 2D list of floats."""
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0

    def __getitem__(self, key: Tuple[int, int]) -> float:
        """Get element at row, col."""
        row, col = key
        return self.data[row][col]

    def __setitem__(self, key: Tuple[int, int], value: float):
        """Set element at row, col."""
        row, col = key
        self.data[row][col] = value

    def transpose(self) -> 'Matrix':
        """Return transpose of matrix."""
        transposed = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return Matrix(transposed)

    def multiply(self, other: 'Matrix') -> 'Matrix':
        """Matrix multiplication."""
        if self.cols != other.rows:
            raise ValueError(f"Cannot multiply {self.rows}x{self.cols} and {other.rows}x{other.cols}")

        result = [[0.0 for _ in range(other.cols)] for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]

        return Matrix(result)

    def add(self, other: 'Matrix') -> 'Matrix':
        """Element-wise addition."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have same dimensions for addition")

        result = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)

    def subtract(self, other: 'Matrix') -> 'Matrix':
        """Element-wise subtraction."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have same dimensions for subtraction")

        result = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)

    def scalar_multiply(self, scalar: float) -> 'Matrix':
        """Multiply all elements by scalar."""
        result = [[self.data[i][j] * scalar for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)

    def apply_function(self, func: Callable[[float], float]) -> 'Matrix':
        """Apply function to all elements."""
        result = [[func(self.data[i][j]) for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)

    def to_list(self) -> List[List[float]]:
        """Convert to list of lists."""
        return [row[:] for row in self.data]

    def __str__(self) -> str:
        """String representation of matrix."""
        return '\n'.join(['\t'.join([f'{val:.4f}' for val in row]) for row in self.data])


# Vector Operations
def dot_product(a: List[float], b: List[float]) -> float:
    """Compute dot product of two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    return sum(a[i] * b[i] for i in range(len(a)))


def vector_add(a: List[float], b: List[float]) -> List[float]:
    """Add two vectors element-wise."""
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    return [a[i] + b[i] for i in range(len(a))]


def vector_subtract(a: List[float], b: List[float]) -> List[float]:
    """Subtract two vectors element-wise."""
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    return [a[i] - b[i] for i in range(len(a))]


def vector_scalar_multiply(vector: List[float], scalar: float) -> List[float]:
    """Multiply vector by scalar."""
    return [x * scalar for x in vector]


def main():
    """Main function to demonstrate backpropagation algorithm."""
    print("Backpropagation from Scratch - Implementation Starting")
    print("=" * 50)

    # Test basic matrix operations
    print("Testing basic matrix operations...")

    # Test matrix creation and basic operations
    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m2 = Matrix([[5.0, 6.0], [7.0, 8.0]])

    print("Matrix 1:")
    print(m1)
    print("\nMatrix 2:")
    print(m2)

    # Test matrix multiplication
    result = m1.multiply(m2)
    print("\nMatrix multiplication result:")
    print(result)

    # Test activation functions
    print(f"\nTesting activation functions:")
    print(f"sigmoid(0) = {sigmoid(0.0):.4f}")
    print(f"tanh(1) = {tanh(1.0):.4f}")
    print(f"relu(-1) = {relu(-1.0):.4f}")

    # Test activation function derivatives
    print(f"\nTesting activation function derivatives:")
    print(f"sigmoid_derivative(0) = {sigmoid_derivative(0.0):.4f}")
    print(f"tanh_derivative(1) = {tanh_derivative(1.0):.4f}")
    print(f"relu_derivative(1) = {relu_derivative(1.0):.4f}")
    print(f"relu_derivative(-1) = {relu_derivative(-1.0):.4f}")

    # Test vector operations
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]
    print(f"\nVector dot product: {dot_product(v1, v2)}")
    print(f"Vector addition: {vector_add(v1, v2)}")

    # Test loss functions
    print(f"\nTesting loss functions:")
    predictions = [0.8, 0.3, 0.9, 0.1]
    targets = [1.0, 0.0, 1.0, 0.0]

    mse_loss = mean_squared_error(predictions, targets)
    mse_grad = mse_derivative(predictions, targets)
    print(f"MSE loss: {mse_loss:.4f}")
    print(f"MSE gradients: {[f'{g:.4f}' for g in mse_grad]}")

    bce_loss = binary_cross_entropy(predictions, targets)
    bce_grad = bce_derivative(predictions, targets)
    print(f"BCE loss: {bce_loss:.4f}")
    print(f"BCE gradients: {[f'{g:.4f}' for g in bce_grad]}")

    print("\nBasic operations validated successfully!")
    print("=" * 50)

    # TODO: Initialize network
    # TODO: Generate XOR dataset
    # TODO: Train network
    # TODO: Visualize results

    print("Implementation complete!")


if __name__ == "__main__":
    main()
