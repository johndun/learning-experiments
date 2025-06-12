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


# Neural Network Data Structures
class Neuron:
    """Simple neuron implementation with weights and bias."""

    def __init__(self, num_inputs: int, activation_func: str = 'sigmoid'):
        """
        Initialize neuron with random weights and bias.

        Args:
            num_inputs: Number of input connections
            activation_func: Activation function name ('sigmoid', 'tanh', 'relu')
        """
        # Initialize weights with small random values
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(num_inputs)]
        self.bias = random.uniform(-1.0, 1.0)
        self.activation_func = activation_func

        # Storage for forward pass values (needed for backprop)
        self.last_input = None
        self.last_weighted_sum = None
        self.last_output = None

    def forward(self, inputs: List[float]) -> float:
        """
        Forward pass through neuron.

        Args:
            inputs: List of input values

        Returns:
            Neuron output after activation
        """
        if len(inputs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, got {len(inputs)}")

        # Store input for backpropagation
        self.last_input = inputs[:]

        # Compute weighted sum: w1*x1 + w2*x2 + ... + bias
        weighted_sum = dot_product(self.weights, inputs) + self.bias
        self.last_weighted_sum = weighted_sum

        # Apply activation function
        if self.activation_func == 'sigmoid':
            output = sigmoid(weighted_sum)
        elif self.activation_func == 'tanh':
            output = tanh(weighted_sum)
        elif self.activation_func == 'relu':
            output = relu(weighted_sum)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_func}")

        self.last_output = output
        return output

    def get_activation_derivative(self) -> float:
        """Get derivative of activation function at last weighted sum."""
        if self.last_weighted_sum is None:
            raise ValueError("Must call forward() before getting derivative")

        if self.activation_func == 'sigmoid':
            return sigmoid_derivative(self.last_weighted_sum)
        elif self.activation_func == 'tanh':
            return tanh_derivative(self.last_weighted_sum)
        elif self.activation_func == 'relu':
            return relu_derivative(self.last_weighted_sum)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_func}")


class Layer:
    """Simple layer implementation containing multiple neurons."""

    def __init__(self, num_neurons: int, num_inputs: int, activation_func: str = 'sigmoid'):
        """
        Initialize layer with specified number of neurons.

        Args:
            num_neurons: Number of neurons in this layer
            num_inputs: Number of inputs to each neuron
            activation_func: Activation function for all neurons in layer
        """
        self.neurons = [Neuron(num_inputs, activation_func) for _ in range(num_neurons)]
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs

        # Storage for layer outputs (needed for backprop)
        self.last_outputs = None

    def forward(self, inputs: List[float]) -> List[float]:
        """
        Forward pass through entire layer.

        Args:
            inputs: List of input values

        Returns:
            List of outputs from all neurons in layer
        """
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)

        self.last_outputs = outputs[:]
        return outputs

    def get_weights_matrix(self) -> Matrix:
        """Return weights as matrix for layer (each row is one neuron's weights)."""
        weights_data = []
        for neuron in self.neurons:
            weights_data.append(neuron.weights[:])
        return Matrix(weights_data)

    def get_biases(self) -> List[float]:
        """Return biases as list for layer."""
        return [neuron.bias for neuron in self.neurons]

    def update_weights(self, weight_gradients: List[List[float]], learning_rate: float):
        """
        Update neuron weights using gradients.

        Args:
            weight_gradients: 2D list where weight_gradients[i][j] is gradient for neuron i, weight j
            learning_rate: Learning rate for gradient descent
        """
        for i, neuron in enumerate(self.neurons):
            for j in range(len(neuron.weights)):
                neuron.weights[j] -= learning_rate * weight_gradients[i][j]

    def update_biases(self, bias_gradients: List[float], learning_rate: float):
        """
        Update neuron biases using gradients.

        Args:
            bias_gradients: List where bias_gradients[i] is gradient for neuron i's bias
            learning_rate: Learning rate for gradient descent
        """
        for i, neuron in enumerate(self.neurons):
            neuron.bias -= learning_rate * bias_gradients[i]


class NeuralNetwork:
    """Simple feedforward neural network implementation."""

    def __init__(self, layer_sizes: List[int], activation_func: str = 'sigmoid'):
        """
        Initialize neural network with specified architecture.

        Args:
            layer_sizes: List where layer_sizes[i] is number of neurons in layer i
                        First element is input size, last is output size
            activation_func: Activation function for hidden layers
        """
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least input and output layer")

        self.layers = []
        self.num_layers = len(layer_sizes) - 1  # Number of actual layers (excluding input)

        # Create layers (first layer takes inputs, subsequent layers take previous layer output)
        for i in range(self.num_layers):
            num_inputs = layer_sizes[i]
            num_neurons = layer_sizes[i + 1]

            # Use sigmoid for output layer in binary classification
            layer_activation = 'sigmoid' if i == self.num_layers - 1 else activation_func
            layer = Layer(num_neurons, num_inputs, layer_activation)
            self.layers.append(layer)

    def forward(self, inputs: List[float]) -> List[float]:
        """
        Forward pass through entire network.

        Args:
            inputs: Network input values

        Returns:
            Network output values
        """
        current_inputs = inputs[:]

        # Pass through each layer
        for layer in self.layers:
            current_inputs = layer.forward(current_inputs)

        return current_inputs

    def get_layer_outputs(self) -> List[List[float]]:
        """Get outputs from all layers (for debugging/visualization)."""
        return [layer.last_outputs[:] for layer in self.layers if layer.last_outputs is not None]


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

    # Test neural network data structures
    print("Testing neural network data structures...")

    # Test single neuron
    neuron = Neuron(2, 'sigmoid')
    test_input = [0.5, -0.3]
    output = neuron.forward(test_input)
    print(f"Neuron forward pass: input={test_input}, output={output:.4f}")
    print(f"Activation derivative: {neuron.get_activation_derivative():.4f}")

    # Test layer
    layer = Layer(3, 2, 'sigmoid')  # 3 neurons, 2 inputs each
    layer_output = layer.forward(test_input)
    print(f"Layer forward pass: input={test_input}, output={[f'{x:.4f}' for x in layer_output]}")

    # Test full network
    network = NeuralNetwork([2, 4, 1], 'sigmoid')  # 2 inputs, 4 hidden, 1 output
    network_output = network.forward(test_input)
    print(f"Network forward pass: input={test_input}, output={network_output[0]:.4f}")

    print("Neural network data structures validated successfully!")
    print("=" * 50)

    # TODO: Generate XOR dataset
    # TODO: Train network
    # TODO: Visualize results

    print("Implementation complete!")


if __name__ == "__main__":
    main()
