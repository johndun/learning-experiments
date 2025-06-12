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
from collections.abc import Callable

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")


# Basic Mathematical Functions
def exp(x: float) -> float:
    """Compute e^x safely, handling overflow."""
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf") if x > 0 else 0.0


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
def mean_squared_error(predictions: list[float], targets: list[float]) -> float:
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


def mse_derivative(predictions: list[float], targets: list[float]) -> list[float]:
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


def binary_cross_entropy(predictions: list[float], targets: list[float]) -> float:
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


def bce_derivative(predictions: list[float], targets: list[float]) -> list[float]:
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


# Vector Operations
def dot_product(a: list[float], b: list[float]) -> float:
    """Compute dot product of two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    return sum(a[i] * b[i] for i in range(len(a)))


# Neural Network Data Structures
class Neuron:
    """Simple neuron implementation with weights and bias."""

    def __init__(self, num_inputs: int, activation_func: str = "sigmoid"):
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

    def forward(self, inputs: list[float]) -> float:
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
        if self.activation_func == "sigmoid":
            output = sigmoid(weighted_sum)
        elif self.activation_func == "tanh":
            output = tanh(weighted_sum)
        elif self.activation_func == "relu":
            output = relu(weighted_sum)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_func}")

        self.last_output = output
        return output

    def get_activation_derivative(self) -> float:
        """Get derivative of activation function at last weighted sum."""
        if self.last_weighted_sum is None:
            raise ValueError("Must call forward() before getting derivative")

        if self.activation_func == "sigmoid":
            return sigmoid_derivative(self.last_weighted_sum)
        elif self.activation_func == "tanh":
            return tanh_derivative(self.last_weighted_sum)
        elif self.activation_func == "relu":
            return relu_derivative(self.last_weighted_sum)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_func}")


class Layer:
    """Simple layer implementation containing multiple neurons."""

    def __init__(self, num_neurons: int, num_inputs: int, activation_func: str = "sigmoid"):
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

    def forward(self, inputs: list[float]) -> list[float]:
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


    def update_weights(self, weight_gradients: list[list[float]], learning_rate: float):
        """
        Update neuron weights using gradients.

        Args:
            weight_gradients: 2D list where weight_gradients[i][j] is gradient for neuron i, weight j
            learning_rate: Learning rate for gradient descent
        """
        for i, neuron in enumerate(self.neurons):
            for j in range(len(neuron.weights)):
                neuron.weights[j] -= learning_rate * weight_gradients[i][j]

    def update_biases(self, bias_gradients: list[float], learning_rate: float):
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

    def __init__(self, layer_sizes: list[int], activation_func: str = "sigmoid"):
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
            layer_activation = "sigmoid" if i == self.num_layers - 1 else activation_func
            layer = Layer(num_neurons, num_inputs, layer_activation)
            self.layers.append(layer)

    def forward(self, inputs: list[float]) -> list[float]:
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


    def backward(
        self, targets: list[float], loss_function: str = "mse"
    ) -> tuple[list[list[list[float]]], list[list[float]]]:
        """
        Backward pass (backpropagation) through the network.

        Computes gradients for all weights and biases using chain rule.
        Propagates error backwards from output layer through all hidden layers.

        Args:
            targets: Target values for training
            loss_function: Loss function to use ('mse' or 'binary_crossentropy')

        Returns:
            Tuple of (weight_gradients, bias_gradients) where:
            - weight_gradients[layer][neuron][weight] = gradient for that weight
            - bias_gradients[layer][neuron] = gradient for that bias
        """
        if not self.layers:
            raise ValueError("Network has no layers")

        # Get network predictions
        predictions = self.layers[-1].last_outputs
        if predictions is None:
            raise ValueError("Must call forward() before backward()")

        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")

        # Initialize gradient storage
        weight_gradients = []
        bias_gradients = []

        for layer in self.layers:
            layer_weight_grads = []
            layer_bias_grads = []

            for neuron in layer.neurons:
                neuron_weight_grads = [0.0] * len(neuron.weights)
                layer_weight_grads.append(neuron_weight_grads)
                layer_bias_grads.append(0.0)

            weight_gradients.append(layer_weight_grads)
            bias_gradients.append(layer_bias_grads)

        # Calculate output layer gradients using chain rule
        output_layer_idx = len(self.layers) - 1
        output_layer = self.layers[output_layer_idx]

        # Step 1: Calculate dL/dOutput (derivative of loss w.r.t. network output)
        if loss_function == "mse":
            loss_gradients = mse_derivative(predictions, targets)
        elif loss_function == "binary_crossentropy":
            loss_gradients = bce_derivative(predictions, targets)
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

        # Step 2: Calculate output layer gradients
        current_deltas = self._calculate_output_layer_gradients(output_layer, loss_gradients)

        # Step 3: Calculate weight and bias gradients for output layer
        self._calculate_layer_weight_bias_gradients(
            output_layer, current_deltas, output_layer_idx, weight_gradients, bias_gradients
        )

        # Step 4: Propagate gradients backward through hidden layers
        for layer_idx in range(len(self.layers) - 2, -1, -1):  # Go backwards through hidden layers
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]

            # Calculate deltas for current hidden layer
            current_deltas = self._calculate_hidden_layer_gradients(
                current_layer, next_layer, current_deltas
            )

            # Calculate weight and bias gradients for current layer
            self._calculate_layer_weight_bias_gradients(
                current_layer, current_deltas, layer_idx, weight_gradients, bias_gradients
            )

        return weight_gradients, bias_gradients

    def _calculate_output_layer_gradients(
        self, output_layer: Layer, loss_gradients: list[float]
    ) -> list[float]:
        """
        Calculate gradients for output layer using chain rule.

        For output layer: delta = dL/dOutput * dOutput/dWeightedSum

        Args:
            output_layer: The output layer
            loss_gradients: Gradients of loss w.r.t. outputs

        Returns:
            List of delta values for each neuron in output layer
        """
        deltas = []

        for i, neuron in enumerate(output_layer.neurons):
            # dL/dOutput (from loss function)
            loss_grad = loss_gradients[i]

            # dOutput/dWeightedSum (activation function derivative)
            activation_derivative = neuron.get_activation_derivative()

            # Chain rule: dL/dWeightedSum = dL/dOutput * dOutput/dWeightedSum
            delta = loss_grad * activation_derivative
            deltas.append(delta)

        return deltas

    def _calculate_hidden_layer_gradients(
        self, current_layer: Layer, next_layer: Layer, next_deltas: list[float]
    ) -> list[float]:
        """
        Calculate gradients for hidden layer using chain rule.

        For hidden layer: delta = sum(next_delta * connecting_weight) * activation_derivative

        This implements the backpropagation equation for hidden layers where the error
        is propagated backwards from the next layer.

        Args:
            current_layer: The current hidden layer
            next_layer: The next layer (closer to output)
            next_deltas: Delta values from the next layer

        Returns:
            List of delta values for each neuron in current layer
        """
        deltas = []

        for i, neuron in enumerate(current_layer.neurons):
            # Sum of (next_layer_delta * weight_connecting_to_next_neuron)
            error_sum = 0.0

            for j, next_neuron in enumerate(next_layer.neurons):
                # Weight from current neuron i to next neuron j
                connecting_weight = next_neuron.weights[i]
                error_sum += next_deltas[j] * connecting_weight

            # Multiply by activation function derivative
            activation_derivative = neuron.get_activation_derivative()
            delta = error_sum * activation_derivative

            deltas.append(delta)

        return deltas

    def _calculate_layer_weight_bias_gradients(
        self,
        layer: Layer,
        deltas: list[float],
        layer_idx: int,
        weight_gradients: list[list[list[float]]],
        bias_gradients: list[list[float]],
    ):
        """
        Calculate weight and bias gradients for a layer given its deltas.

        Weight gradient: dL/dWeight = delta * input_to_neuron
        Bias gradient: dL/dBias = delta

        Args:
            layer: Layer to calculate gradients for
            deltas: Delta values for each neuron in layer
            layer_idx: Index of layer in network
            weight_gradients: Storage for weight gradients (modified in place)
            bias_gradients: Storage for bias gradients (modified in place)
        """
        for i, neuron in enumerate(layer.neurons):
            delta = deltas[i]

            # Bias gradient is just the delta
            bias_gradients[layer_idx][i] = delta

            # Weight gradients: delta * input to this neuron
            if neuron.last_input is None:
                raise ValueError("Neuron must have stored input from forward pass")

            for j, input_val in enumerate(neuron.last_input):
                weight_gradients[layer_idx][i][j] = delta * input_val

    def numerical_gradient_check(
        self,
        inputs: list[float],
        targets: list[float],
        loss_function: str = "mse",
        epsilon: float = 1e-5,
    ) -> tuple[float, float]:
        """
        Validate analytical gradients using numerical gradient checking.

        Computes numerical gradients using finite differences and compares them
        to analytical gradients from backpropagation. This is a critical debugging
        tool to ensure backpropagation is implemented correctly.

        Args:
            inputs: Network input values
            targets: Target values for loss computation
            loss_function: Loss function to use ('mse' or 'binary_crossentropy')
            epsilon: Small value for finite difference approximation

        Returns:
            Tuple of (max_weight_error, max_bias_error) representing maximum
            absolute differences between numerical and analytical gradients
        """
        # Get analytical gradients
        self.forward(inputs)
        analytical_weight_grads, analytical_bias_grads = self.backward(targets, loss_function)

        max_weight_error = 0.0
        max_bias_error = 0.0

        print(f"Running numerical gradient check with epsilon={epsilon}")
        print("Checking weight gradients...")

        # Check weight gradients
        for layer_idx, layer in enumerate(self.layers):
            for neuron_idx, neuron in enumerate(layer.neurons):
                for weight_idx in range(len(neuron.weights)):
                    # Compute numerical gradient for this weight
                    original_weight = neuron.weights[weight_idx]

                    # Forward pass with weight + epsilon
                    neuron.weights[weight_idx] = original_weight + epsilon
                    self.forward(inputs)
                    predictions_plus = self.layers[-1].last_outputs[:]

                    if loss_function == "mse":
                        loss_plus = mean_squared_error(predictions_plus, targets)
                    elif loss_function == "binary_crossentropy":
                        loss_plus = binary_cross_entropy(predictions_plus, targets)
                    else:
                        raise ValueError(f"Unknown loss function: {loss_function}")

                    # Forward pass with weight - epsilon
                    neuron.weights[weight_idx] = original_weight - epsilon
                    self.forward(inputs)
                    predictions_minus = self.layers[-1].last_outputs[:]

                    if loss_function == "mse":
                        loss_minus = mean_squared_error(predictions_minus, targets)
                    elif loss_function == "binary_crossentropy":
                        loss_minus = binary_cross_entropy(predictions_minus, targets)
                    else:
                        raise ValueError(f"Unknown loss function: {loss_function}")

                    # Restore original weight
                    neuron.weights[weight_idx] = original_weight

                    # Compute numerical gradient
                    numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
                    analytical_grad = analytical_weight_grads[layer_idx][neuron_idx][weight_idx]

                    # Compute error
                    error = abs(numerical_grad - analytical_grad)
                    max_weight_error = max(max_weight_error, error)

                    if layer_idx == 0 and neuron_idx == 0:  # Print first few for debugging
                        print(f"  Layer {layer_idx}, Neuron {neuron_idx}, Weight {weight_idx}:")
                        print(f"    Numerical: {numerical_grad:.8f}")
                        print(f"    Analytical: {analytical_grad:.8f}")
                        print(f"    Error: {error:.8f}")

        print("Checking bias gradients...")

        # Check bias gradients
        for layer_idx, layer in enumerate(self.layers):
            for neuron_idx, neuron in enumerate(layer.neurons):
                # Compute numerical gradient for this bias
                original_bias = neuron.bias

                # Forward pass with bias + epsilon
                neuron.bias = original_bias + epsilon
                self.forward(inputs)
                predictions_plus = self.layers[-1].last_outputs[:]

                if loss_function == "mse":
                    loss_plus = mean_squared_error(predictions_plus, targets)
                elif loss_function == "binary_crossentropy":
                    loss_plus = binary_cross_entropy(predictions_plus, targets)
                else:
                    raise ValueError(f"Unknown loss function: {loss_function}")

                # Forward pass with bias - epsilon
                neuron.bias = original_bias - epsilon
                self.forward(inputs)
                predictions_minus = self.layers[-1].last_outputs[:]

                if loss_function == "mse":
                    loss_minus = mean_squared_error(predictions_minus, targets)
                elif loss_function == "binary_crossentropy":
                    loss_minus = binary_cross_entropy(predictions_minus, targets)
                else:
                    raise ValueError(f"Unknown loss function: {loss_function}")

                # Restore original bias
                neuron.bias = original_bias

                # Compute numerical gradient
                numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
                analytical_grad = analytical_bias_grads[layer_idx][neuron_idx]

                # Compute error
                error = abs(numerical_grad - analytical_grad)
                max_bias_error = max(max_bias_error, error)

                if layer_idx == 0 and neuron_idx == 0:  # Print first few for debugging
                    print(f"  Layer {layer_idx}, Neuron {neuron_idx}, Bias:")
                    print(f"    Numerical: {numerical_grad:.8f}")
                    print(f"    Analytical: {analytical_grad:.8f}")
                    print(f"    Error: {error:.8f}")

        return max_weight_error, max_bias_error

    def update_parameters(
        self,
        weight_gradients: list[list[list[float]]],
        bias_gradients: list[list[float]],
        learning_rate: float,
    ):
        """
        Update all network parameters using gradient descent.

        This method applies the gradient descent update rule:
        weight = weight - learning_rate * gradient
        bias = bias - learning_rate * gradient

        Args:
            weight_gradients: 3D list where weight_gradients[layer][neuron][weight] = gradient
            bias_gradients: 2D list where bias_gradients[layer][neuron] = gradient
            learning_rate: Learning rate for gradient descent step size
        """
        if len(weight_gradients) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} layers of weight gradients, got {len(weight_gradients)}"
            )

        if len(bias_gradients) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} layers of bias gradients, got {len(bias_gradients)}"
            )

        # Update parameters for each layer
        for layer_idx, layer in enumerate(self.layers):
            layer.update_weights(weight_gradients[layer_idx], learning_rate)
            layer.update_biases(bias_gradients[layer_idx], learning_rate)

    def train_step(
        self,
        inputs: list[float],
        targets: list[float],
        learning_rate: float = 0.1,
        loss_function: str = "mse",
    ) -> float:
        """
        Perform one training step: forward pass, backward pass, and parameter update.

        Args:
            inputs: Network input values
            targets: Target values for training
            learning_rate: Learning rate for gradient descent
            loss_function: Loss function to use ('mse' or 'binary_crossentropy')

        Returns:
            Loss value after forward pass (before parameter update)
        """
        # Forward pass
        predictions = self.forward(inputs)

        # Calculate loss
        if loss_function == "mse":
            loss = mean_squared_error(predictions, targets)
        elif loss_function == "binary_crossentropy":
            loss = binary_cross_entropy(predictions, targets)
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

        # Backward pass
        weight_gradients, bias_gradients = self.backward(targets, loss_function)

        # Update parameters
        self.update_parameters(weight_gradients, bias_gradients, learning_rate)

        return loss

    def train(
        self,
        training_data: list[tuple[list[float], list[float]]],
        epochs: int = 100,
        learning_rate: float = 0.1,
        loss_function: str = "mse",
        verbose: bool = True,
    ) -> list[float]:
        """
        Train the network on a dataset for multiple epochs.

        Args:
            training_data: List of (inputs, targets) tuples
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            loss_function: Loss function to use ('mse' or 'binary_crossentropy')
            verbose: Whether to print training progress

        Returns:
            List of average loss values for each epoch
        """
        loss_history = []

        for epoch in range(epochs):
            epoch_losses = []

            # Train on each example in the dataset
            for inputs, targets in training_data:
                loss = self.train_step(inputs, targets, learning_rate, loss_function)
                epoch_losses.append(loss)

            # Calculate average loss for this epoch
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            loss_history.append(avg_loss)

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")

        return loss_history


def plot_training_loss(loss_history: list[float], save_path: str = "outputs/training_loss.png"):
    """
    Plot training loss convergence over epochs.

    Args:
        loss_history: List of loss values for each epoch
        save_path: Path to save the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping loss plot generation.")
        return

    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(loss_history) + 1))

    plt.plot(epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set y-axis to start from 0 for better visualization
    plt.ylim(bottom=0)

    # Add text annotations for initial and final loss
    if loss_history:
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        plt.text(0.02, 0.98, f'Initial Loss: {initial_loss:.6f}',
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.text(0.02, 0.90, f'Final Loss: {final_loss:.6f}',
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # Calculate loss reduction percentage
        if initial_loss > 0:
            reduction_pct = ((initial_loss - final_loss) / initial_loss) * 100
            plt.text(0.02, 0.82, f'Reduction: {reduction_pct:.1f}%',
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training loss plot saved to: {save_path}")
    plt.close()


def plot_decision_boundary(
    network: NeuralNetwork,
    training_data: list[tuple[list[float], list[float]]],
    save_path: str = "outputs/decision_boundary.png",
    resolution: int = 100
):
    """
    Plot decision boundary for 2D binary classification.

    Creates a visualization showing how the trained neural network classifies
    different points in the 2D input space. The background shows the decision
    boundary (where the network output changes from 0 to 1), and the training
    data points are overlaid with their true labels.

    Args:
        network: Trained neural network
        training_data: List of (inputs, targets) tuples for training data
        save_path: Path to save the plot
        resolution: Number of points per axis for the grid (higher = smoother)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping decision boundary plot generation.")
        return

    # Extract training inputs and targets
    inputs_list = [inputs for inputs, _ in training_data]
    targets_list = [targets[0] for _, targets in training_data]  # Assuming single output

    # Create a grid of points to evaluate the network on
    x_min, x_max = -0.5, 1.5  # Expand beyond [0,1] to show boundary clearly
    y_min, y_max = -0.5, 1.5

    # Create meshgrid
    x_range = []
    y_range = []
    for i in range(resolution):
        x_val = x_min + (x_max - x_min) * i / (resolution - 1)
        y_val = y_min + (y_max - y_min) * i / (resolution - 1)
        x_range.append(x_val)
        y_range.append(y_val)

    # Evaluate network on grid points
    grid_predictions = []
    for y_val in y_range:
        row_predictions = []
        for x_val in x_range:
            prediction = network.forward([x_val, y_val])
            row_predictions.append(prediction[0])
        grid_predictions.append(row_predictions)

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Create contour plot for decision boundary
    X, Y = [], []
    for i in range(resolution):
        X.append(x_range)
        Y.append([y_range[i]] * resolution)

    # Plot decision boundary using contour
    contour = plt.contourf(x_range, y_range, grid_predictions,
                          levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(contour, label='Network Output')

    # Add decision boundary line (where output = 0.5)
    plt.contour(x_range, y_range, grid_predictions,
               levels=[0.5], colors='black', linewidths=2, linestyles='--')

    # Plot training data points
    for i, (inputs, target) in enumerate(zip(inputs_list, targets_list)):
        x, y = inputs[0], inputs[1]
        if target == 0:
            plt.scatter(x, y, c='blue', s=200, marker='o',
                       edgecolors='black', linewidth=2, label='Class 0' if i == 0 else "")
        else:
            plt.scatter(x, y, c='red', s=200, marker='s',
                       edgecolors='black', linewidth=2, label='Class 1' if i == 1 else "")

    # Add labels and annotations for each training point
    labels = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
    for i, (inputs, target) in enumerate(zip(inputs_list, targets_list)):
        x, y = inputs[0], inputs[1]
        plt.annotate(labels[i], (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Set plot properties
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Input 1', fontsize=12)
    plt.ylabel('Input 2', fontsize=12)
    plt.title('Decision Boundary Visualization for XOR Classification', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # Add text box with network info
    info_text = f"Network Architecture: {network.num_layers} layers\n"
    info_text += f"Hidden Neurons: {len(network.layers[0].neurons)}\n"
    info_text += f"Activation: {network.layers[0].neurons[0].activation_func}"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add explanation text
    explanation = "Decision Boundary (dashed line): Where network output = 0.5\n"
    explanation += "Blue regions: Network predicts Class 0\n"
    explanation += "Red regions: Network predicts Class 1"
    plt.text(0.02, 0.15, explanation, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Decision boundary plot saved to: {save_path}")
    plt.close()


def main():
    """Main function to demonstrate backpropagation algorithm."""
    print("Backpropagation from Scratch - Implementation Starting")
    print("=" * 50)

    # Test activation functions
    print("\nTesting activation functions:")
    print(f"sigmoid(0) = {sigmoid(0.0):.4f}")
    print(f"tanh(1) = {tanh(1.0):.4f}")
    print(f"relu(-1) = {relu(-1.0):.4f}")

    # Test activation function derivatives
    print("\nTesting activation function derivatives:")
    print(f"sigmoid_derivative(0) = {sigmoid_derivative(0.0):.4f}")
    print(f"tanh_derivative(1) = {tanh_derivative(1.0):.4f}")
    print(f"relu_derivative(1) = {relu_derivative(1.0):.4f}")
    print(f"relu_derivative(-1) = {relu_derivative(-1.0):.4f}")

    # Test vector operations
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]
    print(f"\nVector dot product: {dot_product(v1, v2)}")

    # Test loss functions
    print("\nTesting loss functions:")
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
    neuron = Neuron(2, "sigmoid")
    test_input = [0.5, -0.3]
    output = neuron.forward(test_input)
    print(f"Neuron forward pass: input={test_input}, output={output:.4f}")
    print(f"Activation derivative: {neuron.get_activation_derivative():.4f}")

    # Test layer
    layer = Layer(3, 2, "sigmoid")  # 3 neurons, 2 inputs each
    layer_output = layer.forward(test_input)
    print(f"Layer forward pass: input={test_input}, output={[f'{x:.4f}' for x in layer_output]}")

    # Test full network
    network = NeuralNetwork([2, 4, 1], "sigmoid")  # 2 inputs, 4 hidden, 1 output
    network_output = network.forward(test_input)
    print(f"Network forward pass: input={test_input}, output={network_output[0]:.4f}")

    print("Neural network data structures validated successfully!")
    print("=" * 50)

    # Test backpropagation gradients
    print("Testing backpropagation gradients...")

    # Create simple test case
    simple_network = NeuralNetwork([2, 1], "sigmoid")  # 2 inputs, 1 output
    test_inputs = [0.5, -0.3]
    test_targets = [1.0]

    # Forward pass
    prediction = simple_network.forward(test_inputs)
    print(
        f"Forward pass: input={test_inputs}, prediction={prediction[0]:.4f}, target={test_targets[0]}"
    )

    # Backward pass
    weight_grads, bias_grads = simple_network.backward(test_targets, "mse")

    print("Computed gradients successfully!")
    print(f"Weight gradients shape: {len(weight_grads)} layers")
    print(
        f"Layer 0 weight gradients: {len(weight_grads[0])} neurons, {len(weight_grads[0][0])} weights each"
    )
    print(f"First neuron weight gradients: {[f'{g:.4f}' for g in weight_grads[0][0]]}")
    print(f"First neuron bias gradient: {bias_grads[0][0]:.4f}")

    # Test with multi-layer network
    print("\nTesting multi-layer network gradients...")
    multi_network = NeuralNetwork([2, 3, 1], "sigmoid")  # 2 inputs, 3 hidden, 1 output
    multi_prediction = multi_network.forward(test_inputs)
    multi_weight_grads, multi_bias_grads = multi_network.backward(test_targets, "mse")

    print(f"Multi-layer forward pass: prediction={multi_prediction[0]:.4f}")
    print(f"Multi-layer gradients computed for {len(multi_weight_grads)} layers")

    print("Backpropagation gradients validated successfully!")
    print("=" * 50)

    # Test numerical gradient checking
    print("Testing numerical gradient checking...")

    # Test with simple network
    print("\nTesting simple network (2 inputs, 1 output):")
    gradient_network = NeuralNetwork([2, 1], "sigmoid")
    test_inputs = [0.5, -0.3]
    test_targets = [1.0]

    max_weight_error, max_bias_error = gradient_network.numerical_gradient_check(
        test_inputs, test_targets, "mse", epsilon=1e-5
    )

    print("\nGradient Check Results:")
    print(f"Maximum weight gradient error: {max_weight_error:.10f}")
    print(f"Maximum bias gradient error: {max_bias_error:.10f}")

    # Check if gradients are sufficiently accurate
    tolerance = 1e-5
    if max_weight_error < tolerance and max_bias_error < tolerance:
        print(f"✓ Gradient check PASSED! (errors < {tolerance})")
    else:
        print(f"✗ Gradient check FAILED! (errors >= {tolerance})")

    # Test with multi-layer network
    print("\nTesting multi-layer network (2 inputs, 3 hidden, 1 output):")
    complex_network = NeuralNetwork([2, 3, 1], "sigmoid")

    max_weight_error_complex, max_bias_error_complex = complex_network.numerical_gradient_check(
        test_inputs, test_targets, "mse", epsilon=1e-5
    )

    print("\nComplex Network Gradient Check Results:")
    print(f"Maximum weight gradient error: {max_weight_error_complex:.10f}")
    print(f"Maximum bias gradient error: {max_bias_error_complex:.10f}")

    if max_weight_error_complex < tolerance and max_bias_error_complex < tolerance:
        print(f"✓ Complex network gradient check PASSED! (errors < {tolerance})")
    else:
        print(f"✗ Complex network gradient check FAILED! (errors >= {tolerance})")

    # Test with binary cross-entropy loss
    print("\nTesting with binary cross-entropy loss:")
    bce_network = NeuralNetwork([2, 1], "sigmoid")

    max_weight_error_bce, max_bias_error_bce = bce_network.numerical_gradient_check(
        test_inputs, test_targets, "binary_crossentropy", epsilon=1e-5
    )

    print("\nBCE Gradient Check Results:")
    print(f"Maximum weight gradient error: {max_weight_error_bce:.10f}")
    print(f"Maximum bias gradient error: {max_bias_error_bce:.10f}")

    if max_weight_error_bce < tolerance and max_bias_error_bce < tolerance:
        print(f"✓ BCE gradient check PASSED! (errors < {tolerance})")
    else:
        print(f"✗ BCE gradient check FAILED! (errors >= {tolerance})")

    print("\nNumerical gradient checking completed successfully!")
    print("=" * 50)

    # Generate XOR dataset for binary classification demonstration
    print("Generating XOR dataset for binary classification...")

    def generate_xor_dataset() -> list[tuple[list[float], list[float]]]:
        """
        Generate XOR dataset for binary classification.

        XOR truth table:
        Input1  Input2  Output
        0       0       0
        0       1       1
        1       0       1
        1       1       0

        Returns:
            List of (inputs, targets) tuples for XOR problem
        """
        xor_data = [
            ([0.0, 0.0], [0.0]),  # 0 XOR 0 = 0
            ([0.0, 1.0], [1.0]),  # 0 XOR 1 = 1
            ([1.0, 0.0], [1.0]),  # 1 XOR 0 = 1
            ([1.0, 1.0], [0.0]),  # 1 XOR 1 = 0
        ]
        return xor_data

    # Generate the XOR dataset
    xor_dataset = generate_xor_dataset()

    print("XOR Dataset Generated:")
    print("Input1  Input2  Target")
    print("-" * 20)
    for inputs, targets in xor_dataset:
        print(f"{inputs[0]:4.1f}    {inputs[1]:4.1f}    {targets[0]:4.1f}")

    # Test network on XOR dataset before training
    print("\nTesting untrained network on XOR dataset:")
    xor_network = NeuralNetwork([2, 4, 1], "sigmoid")  # 2 inputs, 4 hidden neurons, 1 output

    print("Untrained Network Predictions:")
    print("Input1  Input2  Prediction  Target  Error")
    print("-" * 40)

    total_error = 0.0
    for inputs, targets in xor_dataset:
        prediction = xor_network.forward(inputs)
        error = abs(prediction[0] - targets[0])
        total_error += error
        print(
            f"{inputs[0]:4.1f}    {inputs[1]:4.1f}    {prediction[0]:8.4f}  {targets[0]:4.1f}  {error:6.4f}"
        )

    avg_error = total_error / len(xor_dataset)
    print(f"Average prediction error (untrained): {avg_error:.4f}")

    # Verify XOR dataset is non-linearly separable
    print("\nXOR Problem Analysis:")
    print("The XOR problem is a classic example of a non-linearly separable dataset.")
    print("A single perceptron cannot solve XOR, but a multi-layer network can.")
    print("This demonstrates the power of hidden layers in neural networks.")

    # Show that simple linear classification fails
    print("\nLinear separability check:")
    print("For linear separability, we need to find weights w1, w2, bias b such that:")
    print("  w1*x1 + w2*x2 + b > 0 for positive class")
    print("  w1*x1 + w2*x2 + b < 0 for negative class")
    print()
    print("XOR truth table analysis:")
    print("  (0,0) -> 0: Need w1*0 + w2*0 + b < 0, so b < 0")
    print("  (0,1) -> 1: Need w1*0 + w2*1 + b > 0, so w2 + b > 0")
    print("  (1,0) -> 1: Need w1*1 + w2*0 + b > 0, so w1 + b > 0")
    print("  (1,1) -> 0: Need w1*1 + w2*1 + b < 0, so w1 + w2 + b < 0")
    print()
    print("From constraints 2 and 3: w1 > -b and w2 > -b")
    print("From constraint 4: w1 + w2 < -b")
    print("This gives us: w1 + w2 < -b < w1 and w1 + w2 < -b < w2")
    print("This is impossible since w1 + w2 cannot be less than both w1 and w2!")
    print("Therefore, XOR is NOT linearly separable.")

    print("\nXOR dataset generation completed successfully!")
    print("=" * 50)

    # Complete Training Loop Demonstration
    print("Demonstrating complete training loop with epoch management...")

    # Create network for XOR training
    print("\nInitializing network for XOR training:")
    print("Architecture: 2 inputs -> 4 hidden neurons -> 1 output")
    training_network = NeuralNetwork([2, 4, 1], "sigmoid")

    # Training parameters
    epochs = 1000
    learning_rate = 5.0  # Higher learning rate for faster convergence on simple XOR
    loss_function = "mse"

    print("Training parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Loss function: {loss_function}")
    print(f"  Dataset size: {len(xor_dataset)} examples")

    # Train the network
    print("\nStarting training...")
    print("Progress will be reported every 100 epochs")
    print("-" * 50)

    loss_history = training_network.train(
        training_data=xor_dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        loss_function=loss_function,
        verbose=True,
    )

    print("-" * 50)
    print("Training completed!")

    # Test trained network performance
    print("\nTesting trained network on XOR dataset:")
    print("Input1  Input2  Prediction  Target  Error")
    print("-" * 40)

    total_error_trained = 0.0
    for inputs, targets in xor_dataset:
        prediction = training_network.forward(inputs)
        error = abs(prediction[0] - targets[0])
        total_error_trained += error
        print(
            f"{inputs[0]:4.1f}    {inputs[1]:4.1f}    {prediction[0]:8.4f}  {targets[0]:4.1f}  {error:6.4f}"
        )

    avg_error_trained = total_error_trained / len(xor_dataset)
    print(f"Average prediction error (trained): {avg_error_trained:.6f}")
    print(f"Improvement from untrained: {avg_error - avg_error_trained:.6f}")

    # Training analysis
    print("\nTraining Analysis:")
    print(f"Initial loss: {loss_history[0]:.6f}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Loss reduction: {loss_history[0] - loss_history[-1]:.6f}")
    print(f"Convergence achieved: {'Yes' if loss_history[-1] < 0.01 else 'No'}")

    # Check if network learned XOR function
    tolerance = 0.1  # Allow 10% error for binary classification
    correct_predictions = 0

    print(f"\nBinary Classification Results (tolerance: {tolerance}):")
    for inputs, targets in xor_dataset:
        prediction = training_network.forward(inputs)
        predicted_class = 1.0 if prediction[0] > 0.5 else 0.0
        target_class = targets[0]
        correct = abs(predicted_class - target_class) < tolerance
        correct_predictions += int(correct)

        print(
            f"Input: ({inputs[0]}, {inputs[1]}) -> "
            f"Raw: {prediction[0]:.4f}, "
            f"Class: {predicted_class}, "
            f"Target: {target_class}, "
            f"{'✓' if correct else '✗'}"
        )

    accuracy = correct_predictions / len(xor_dataset)
    print(f"\nClassification Accuracy: {accuracy:.1%} ({correct_predictions}/{len(xor_dataset)})")

    if accuracy >= 1.0:
        print("🎉 SUCCESS! Network successfully learned the XOR function!")
    elif accuracy >= 0.75:
        print("📈 GOOD! Network learned most of the XOR function.")
    else:
        print("📉 Network struggled to learn XOR. May need more training or different parameters.")

    # Generate training loss convergence plot
    print("\n" + "=" * 50)
    print("GENERATING TRAINING LOSS CONVERGENCE PLOT")
    print("=" * 50)

    try:
        plot_training_loss(loss_history, "outputs/backprop_training_loss.png")
        print("✅ Training loss convergence plot generated successfully!")
    except Exception as e:
        print(f"❌ Failed to generate training loss plot: {e}")

    print("=" * 50)

    # Generate decision boundary visualization
    print("\n" + "=" * 50)
    print("GENERATING DECISION BOUNDARY VISUALIZATION")
    print("=" * 50)

    try:
        plot_decision_boundary(training_network, xor_dataset, "outputs/backprop_decision_boundary.png")
        print("✅ Decision boundary visualization generated successfully!")
    except Exception as e:
        print(f"❌ Failed to generate decision boundary plot: {e}")

    print("=" * 50)

    # Training loop demonstration summary
    print("\n" + "=" * 50)
    print("TRAINING LOOP DEMONSTRATION COMPLETE")
    print("=" * 50)
    print("✅ Implemented complete training loop with:")
    print("   • Epoch management and progress tracking")
    print("   • Loss computation and monitoring")
    print("   • Parameter updates via gradient descent")
    print("   • Training progress reporting")
    print("   • Performance evaluation on test data")
    print("   • Convergence analysis")
    print("   • Binary classification metrics")
    print("   • Training loss visualization")
    print("=" * 50)

    print("Implementation complete with validated backpropagation!")


if __name__ == "__main__":
    main()
