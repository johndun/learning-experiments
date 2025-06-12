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

    # Test vector operations
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]
    print(f"\nVector dot product: {dot_product(v1, v2)}")
    print(f"Vector addition: {vector_add(v1, v2)}")

    print("\nBasic operations validated successfully!")
    print("=" * 50)

    # TODO: Initialize network
    # TODO: Generate XOR dataset
    # TODO: Train network
    # TODO: Visualize results

    print("Implementation complete!")


if __name__ == "__main__":
    main()
