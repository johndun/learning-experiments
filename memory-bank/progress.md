# Single Script Backpropagation Implementation Plan

## Goal
Create a single Python script that implements backpropagation from scratch, demonstrating the complete algorithm without external dependencies (except for basic visualization with matplotlib/numpy).

Do not write separate test files. Validate functionality with inline print statements.

## Script Structure and Implementation

- [x] Create main script file (`scripts/backprop_from_scratch.py`) containing all components
- [x] Implement basic matrix operations and mathematical functions from scratch
- [x] Define activation functions (sigmoid, tanh, ReLU) and their derivatives inline
- [ ] Implement loss functions (mean squared error) with derivatives

## Core Neural Network Components (All in Single Script)

- [ ] Define simple neuron/layer data structures using basic Python data types
- [ ] Implement forward propagation function that processes input through network layers
- [ ] Store intermediate activations during forward pass for backpropagation use
- [ ] Create network initialization function for weights and biases

## Backpropagation Algorithm Implementation

- [ ] Implement gradient calculation for output layer using chain rule
- [ ] Implement gradient calculation for hidden layers propagating error backwards
- [ ] Validate gradients using numerical gradient checking
- [ ] Calculate weight and bias gradients for all layers
- [ ] Implement gradient descent parameter update function

## Training and Demonstration

- [ ] Generate XOR dataset directly in script for binary classification demo
- [ ] Create complete training loop with epoch management
- [ ] Implement simple batch processing for training
- [ ] Add training progress tracking (loss over epochs)

## Visualization and Output

- [ ] Generate training loss convergence plot
- [ ] Create decision boundary visualization for 2D classification
