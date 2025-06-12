# LeNet MNIST Training with Lightning Fabric + Hydra

This script demonstrates modern MLOps practices by combining Lightning Fabric and Hydra for LeNet-5 training on MNIST.

## Features

- **Lightning Fabric**: Device-agnostic training, mixed precision, distributed training support
- **Hydra Configuration**: Flexible YAML-based configuration with CLI overrides
- **Modular Design**: Separate configs for model, training, and data
- **Comprehensive Logging**: Structured logging with configurable levels
- **Visualization**: Training curves, sample predictions, and feature maps

## Usage Examples

### Basic Usage
```bash
python lenet_mnist_fabric_hydra.py
```

### Fast Training Configuration
```bash
python lenet_mnist_fabric_hydra.py training=fast
```

### CLI Overrides
```bash
# Change epochs and learning rate
python lenet_mnist_fabric_hydra.py training.epochs=10 training.learning_rate=0.001

# Enable model compilation
python lenet_mnist_fabric_hydra.py model.compile.enabled=true

# Use full dataset instead of subset
python lenet_mnist_fabric_hydra.py data.subset.enabled=false

# Disable visualizations
python lenet_mnist_fabric_hydra.py visualizations.enabled=false

# Mixed precision training
python lenet_mnist_fabric_hydra.py training.performance.precision=16-mixed
```

### Multiple Overrides
```bash
python lenet_mnist_fabric_hydra.py \
    training=fast \
    training.epochs=8 \
    model.compile.enabled=true \
    data.subset.ratio=0.2
```

## Configuration Structure

```
configs/
├── config.yaml              # Main configuration
├── model/
│   └── lenet5.yaml          # Model architecture
├── training/
│   ├── default.yaml         # Default hyperparameters
│   └── fast.yaml           # Fast training setup
└── data/
    └── mnist.yaml          # Dataset configuration
```

## Key Improvements Over Original Script

1. **Configuration Management**: YAML configs instead of hardcoded values
2. **Device Handling**: Automatic device selection and data loading
3. **Performance**: Mixed precision and compilation support
4. **Experiment Tracking**: Automatic output directory with saved configs
5. **Modularity**: Easy to swap configurations for different experiments

## Output Structure

When you run the script, Hydra creates organized output directories:
```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── config.yaml                          # Saved configuration
        ├── lenet_fabric_training_curves.png     # Training plots
        ├── lenet_fabric_sample_predictions.png  # Sample predictions
        └── lenet_fabric_feature_maps.png        # Feature visualizations
```

## Comparison with Original Script

| Feature | Original | Fabric + Hydra |
|---------|----------|----------------|
| Configuration | Hardcoded dict | YAML configs with CLI overrides |
| Device Management | Manual | Automatic via Fabric |
| Mixed Precision | Manual | Built-in Fabric support |
| Experiment Tracking | Basic | Structured with Hydra |
| Reproducibility | Manual seeds | Config preservation |
| Distributed Training | Not supported | Ready via Fabric |
