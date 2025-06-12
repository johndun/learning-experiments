# learning-experiments

Machine learning experiments implementing neural networks and optimization algorithms from scratch.

## Scripts

### 1. Backpropagation from Scratch (`scripts/backprop_from_scratch.py`)
Complete implementation of neural network backpropagation with:
- Multi-layer feedforward networks
- Activation functions (sigmoid, tanh, ReLU)
- Loss functions (MSE, binary cross-entropy)
- Numerical gradient checking
- XOR problem demonstration
- Training visualizations

### 2. Gradient Descent Demo (`scripts/gradient_descent_demo.py`)
PyTorch-based gradient descent implementation comparing:
- Manual gradient descent
- Built-in PyTorch optimizers (SGD, Adam, RMSprop)
- Convergence analysis and visualization
- Performance comparisons

## Running the Scripts

Execute scripts and save outputs:

```bash
# Run backpropagation demo
python scripts/backprop_from_scratch.py > outputs/backprop_from_scratch_output.txt

# Run gradient descent demo
python scripts/gradient_descent_demo.py > outputs/gradient_descent_demo_output.txt
```

## Generated Outputs

The scripts generate text outputs and visualizations in `outputs/`:

**Text Outputs:**
- `backprop_from_scratch_output.txt` - Detailed training logs and analysis
- `gradient_descent_demo_output.txt` - Optimization comparison results

**Visualizations:**
- `backprop_training_loss.png` - Training loss convergence
- `backprop_decision_boundary.png` - XOR decision boundary
- `gradient_descent_convergence.png` - Convergence comparison
- `gradient_descent_function.png` - Target function plot
- `gradient_descent_with_path.png` - Optimization path
- `optimizer_comparison.png` - Optimizer performance comparison

## Requirements

- Python 3.12+
- PyTorch 2.7+
- NumPy
- Matplotlib
