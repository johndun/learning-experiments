#!/usr/bin/env python3
"""
LeNet MNIST Training Script with Lightning Fabric + Hydra

A modernized implementation of LeNet-5 architecture for MNIST digit classification
using Lightning Fabric for device management and Hydra for configuration management.

Key Features:
- Lightning Fabric for device-agnostic training and performance optimizations
- Hydra for flexible YAML-based configuration management
- LeNet-5 convolutional neural network architecture
- MNIST dataset loading with configurable subsets
- Comprehensive training visualization and analysis
- Support for mixed precision training and model compilation

Usage:
    python lenet_mnist_fabric_hydra.py
    python lenet_mnist_fabric_hydra.py training=fast
    python lenet_mnist_fabric_hydra.py training.epochs=10 training.learning_rate=0.001
    python lenet_mnist_fabric_hydra.py model.compile.enabled=true

Author: Learning Experiments
"""

import os
import time
import logging
from typing import Tuple, List, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.fabric import Fabric

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available. Visualizations will be skipped.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LeNet5(nn.Module):
    """
    LeNet-5 Convolutional Neural Network for MNIST classification.

    Configurable architecture via Hydra config.

    Architecture:
    - Conv2d(1, 6, 5) + ReLU + MaxPool2d(2, 2)
    - Conv2d(6, 16, 5) + ReLU + MaxPool2d(2, 2)
    - Flatten
    - Linear(16*4*4, 120) + ReLU
    - Linear(120, 84) + ReLU
    - Linear(84, 10) + LogSoftmax

    Input: 28x28 grayscale images (MNIST digits)
    Output: 10 class probabilities (digits 0-9)
    """

    def __init__(self, config: DictConfig):
        super(LeNet5, self).__init__()
        self.config = config

        # Feature extraction layers (convolutional)
        self.conv1 = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.conv1.out_channels,
            kernel_size=config.conv1.kernel_size,
            padding=config.conv1.padding
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=config.pooling.kernel_size,
            stride=config.pooling.stride
        )

        self.conv2 = nn.Conv2d(
            in_channels=config.conv1.out_channels,
            out_channels=config.conv2.out_channels,
            kernel_size=config.conv2.kernel_size,
            padding=config.conv2.padding
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=config.pooling.kernel_size,
            stride=config.pooling.stride
        )

        # Calculate flattened size: 28->24->12->8->4, so 16*4*4=256
        conv_output_size = config.conv2.out_channels * 4 * 4

        # Classification layers (fully connected)
        self.fc1 = nn.Linear(conv_output_size, config.fc1.out_features)
        self.fc2 = nn.Linear(config.fc1.out_features, config.fc2.out_features)
        self.fc3 = nn.Linear(config.fc2.out_features, config.num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights based on config."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if self.config.weight_init.method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                if self.config.weight_init.method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LeNet-5.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output tensor of shape (batch_size, 10) with log probabilities
        """
        # First convolutional block
        x = self.conv1(x)           # (batch, 1, 28, 28) -> (batch, 6, 24, 24)
        x = F.relu(x)
        x = self.pool1(x)           # (batch, 6, 24, 24) -> (batch, 6, 12, 12)

        # Second convolutional block
        x = self.conv2(x)           # (batch, 6, 12, 12) -> (batch, 16, 8, 8)
        x = F.relu(x)
        x = self.pool2(x)           # (batch, 16, 8, 8) -> (batch, 16, 4, 4)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)   # (batch, 16, 4, 4) -> (batch, 256)

        # Fully connected layers
        x = self.fc1(x)             # (batch, 256) -> (batch, 120)
        x = F.relu(x)

        x = self.fc2(x)             # (batch, 120) -> (batch, 84)
        x = F.relu(x)

        x = self.fc3(x)             # (batch, 84) -> (batch, 10)

        # Apply log softmax for classification
        return F.log_softmax(x, dim=1)

    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get intermediate feature maps for visualization.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Tuple of (conv1_features, conv2_features)
        """
        # First conv layer features
        conv1_out = F.relu(self.conv1(x))
        conv1_pooled = self.pool1(conv1_out)

        # Second conv layer features
        conv2_out = F.relu(self.conv2(conv1_pooled))

        return conv1_out, conv2_out


def create_data_loaders(config: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create MNIST data loaders based on config.

    Args:
        config: Hydra configuration

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Data transformations
    transform_list = [transforms.ToTensor()]

    if config.data.transforms.normalize:
        transform_list.append(
            transforms.Normalize(
                config.data.transforms.normalize.mean,
                config.data.transforms.normalize.std
            )
        )

    transform = transforms.Compose(transform_list)

    # Download and load datasets
    logger.info("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root=config.data.root,
        train=True,
        download=config.data.download,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=config.data.root,
        train=False,
        download=config.data.download,
        transform=transform
    )

    # Use subset if configured
    if config.data.subset.enabled:
        train_size = int(len(train_dataset) * config.data.subset.ratio)
        test_size = int(len(test_dataset) * config.data.subset.ratio)

        if config.data.subset.shuffle:
            train_indices = torch.randperm(len(train_dataset))[:train_size]
            test_indices = torch.randperm(len(test_dataset))[:test_size]
        else:
            train_indices = torch.arange(train_size)
            test_indices = torch.arange(test_size)

        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

        logger.info(f"Using subset: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    else:
        logger.info(f"Using full dataset: {len(train_dataset)} training samples, {len(test_dataset)} test samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    return train_loader, test_loader


def create_optimizer(model: nn.Module, config: DictConfig) -> optim.Optimizer:
    """Create optimizer based on config."""
    if config.training.optimizer.name.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.optimizer.weight_decay,
            nesterov=config.training.optimizer.nesterov
        )
    elif config.training.optimizer.name.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.optimizer.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer.name}")


def create_criterion(config: DictConfig) -> nn.Module:
    """Create loss function based on config."""
    if config.training.criterion == "nll_loss":
        return nn.NLLLoss()
    elif config.training.criterion == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {config.training.criterion}")


def train_epoch(
    fabric: Fabric,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    config: DictConfig
) -> Tuple[float, float]:
    """
    Train model for one epoch using Lightning Fabric.

    Args:
        fabric: Lightning Fabric instance
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        epoch: Current epoch number
        config: Training configuration

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    logger.info(f"Epoch {epoch + 1} Training:")
    logger.info("-" * 50)

    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass using Fabric
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # Calculate statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Log progress
        if batch_idx % config.logging.log_every_n_batches == 0:
            progress = 100.0 * batch_idx / len(train_loader)
            current_acc = 100.0 * correct / total
            logger.info(f"Batch {batch_idx:3d}/{len(train_loader)} ({progress:5.1f}%) | "
                       f"Loss: {loss.item():.6f} | Acc: {current_acc:.2f}%")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    logger.info(f"Training Results: Loss={avg_loss:.6f}, Accuracy={accuracy:.2f}%")
    return avg_loss, accuracy


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, float]:
    """
    Evaluate model on test dataset.

    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss function

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)

            # Calculate loss
            total_loss += criterion(output, target).item()

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    logger.info(f"Test Results: Loss={avg_loss:.6f}, Accuracy={accuracy:.2f}%")
    return avg_loss, accuracy


def plot_training_curves(
    train_losses: List[float],
    train_accuracies: List[float],
    test_losses: List[float],
    test_accuracies: List[float],
    save_dir: str
):
    """Plot training and validation curves."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Skipping training curves plot.")
        return

    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()

    # Save plot
    loss_path = os.path.join(save_dir, "lenet_fabric_training_curves.png")
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training curves saved to: {loss_path}")
    plt.close()


def plot_sample_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    num_samples: int,
    save_dir: str
):
    """Plot sample predictions from the model."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Skipping sample predictions plot.")
        return

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Get one batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

    # Select samples to display
    num_samples = min(num_samples, images.size(0))

    # Create subplot grid
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle('LeNet-5 Sample Predictions (Fabric + Hydra)', fontsize=16, fontweight='bold')

    for i in range(num_samples):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        # Convert tensor to numpy for plotting
        image = images[i].cpu().squeeze().numpy()
        true_label = labels[i].item()
        pred_label = predictions[i].item()

        # Plot image
        ax.imshow(image, cmap='gray')
        ax.set_title(f'True: {true_label}, Pred: {pred_label}',
                    color='green' if true_label == pred_label else 'red',
                    fontweight='bold')
        ax.axis('off')

    # Hide unused subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.tight_layout()

    # Save plot
    pred_path = os.path.join(save_dir, "lenet_fabric_sample_predictions.png")
    plt.savefig(pred_path, dpi=300, bbox_inches='tight')
    logger.info(f"Sample predictions saved to: {pred_path}")
    plt.close()


def plot_feature_maps(model: nn.Module, test_loader: DataLoader, save_dir: str):
    """Visualize learned feature maps from convolutional layers."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Skipping feature maps plot.")
        return

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Get one sample
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    sample_image = images[0:1]  # Take first image, keep batch dimension
    sample_label = labels[0].item()

    # Get feature maps
    with torch.no_grad():
        conv1_features, conv2_features = model.get_feature_maps(sample_image)

    # Convert to numpy
    original_image = sample_image.cpu().squeeze().numpy()
    conv1_maps = conv1_features.cpu().squeeze().numpy()  # Shape: (6, 24, 24)
    conv2_maps = conv2_features.cpu().squeeze().numpy()  # Shape: (16, 8, 8)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(3, 8, figure=fig)

    # Original image
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original_image, cmap='gray')
    ax_orig.set_title(f'Original Image\n(Label: {sample_label})', fontweight='bold')
    ax_orig.axis('off')

    # Conv1 feature maps (6 features)
    for i in range(6):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(conv1_maps[i], cmap='viridis')
        ax.set_title(f'Conv1 Feature {i+1}', fontsize=10)
        ax.axis('off')

    # Conv2 feature maps (show first 16)
    for i in range(16):
        row = 1 + i // 8
        col = i % 8
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(conv2_maps[i], cmap='viridis')
        ax.set_title(f'Conv2 Feature {i+1}', fontsize=9)
        ax.axis('off')

    plt.suptitle('LeNet-5 Learned Feature Maps (Fabric + Hydra)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save plot
    features_path = os.path.join(save_dir, "lenet_fabric_feature_maps.png")
    plt.savefig(features_path, dpi=300, bbox_inches='tight')
    logger.info(f"Feature maps saved to: {features_path}")
    plt.close()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function with Lightning Fabric and Hydra configuration."""
    logger.info("LeNet-5 MNIST Training with Lightning Fabric + Hydra")
    logger.info("=" * 60)

    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(config))

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)

    # Initialize Lightning Fabric
    fabric = Fabric(
        precision=config.training.performance.precision,
        devices="auto",
        accelerator="auto"
    )
    fabric.launch()

    logger.info(f"Using device: {fabric.device}")
    logger.info(f"Precision: {config.training.performance.precision}")

    # Create output directory using Hydra's working directory
    output_dir = Path.cwd()  # Hydra automatically sets the working directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration to output directory
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)
    logger.info(f"Configuration saved to: {config_path}")

    # Create data loaders
    start_time = time.time()
    train_loader, test_loader = create_data_loaders(config)
    data_time = time.time() - start_time
    logger.info(f"Data loading completed in {data_time:.2f} seconds")

    # Setup data loaders with Fabric
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # Create model
    logger.info("Initializing LeNet-5 model...")
    model = LeNet5(config.model)

    # Print model architecture
    logger.info(f"Model Architecture:\n{model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer and criterion
    optimizer = create_optimizer(model, config)
    criterion = create_criterion(config)

    logger.info(f"Optimizer: {config.training.optimizer.name}")
    logger.info(f"Loss function: {config.training.criterion}")

    # Setup model and optimizer with Fabric
    model, optimizer = fabric.setup(model, optimizer)

    # Apply torch.compile if configured
    if config.training.performance.compile_model or config.model.compile.enabled:
        logger.info("Applying torch.compile optimization...")
        compile_mode = getattr(config.model.compile, 'mode', 'default')
        model = torch.compile(model, mode=compile_mode)
        logger.info("✅ Model compiled successfully!")

    # Training loop
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    training_start_time = time.time()

    logger.info(f"Starting training for {config.training.epochs} epochs...")
    logger.info("=" * 60)

    for epoch in range(config.training.epochs):
        epoch_start_time = time.time()

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            fabric, model, train_loader, optimizer, criterion, epoch, config
        )

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_loss, test_acc = test_model(model, test_loader, criterion)

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
        logger.info("=" * 50)

    total_training_time = time.time() - training_start_time

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total training time: {total_training_time:.2f} seconds")
    logger.info(f"Average time per epoch: {total_training_time/config.training.epochs:.2f} seconds")

    # Final results
    logger.info("\nFinal Training Results:")
    logger.info(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    logger.info(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    logger.info(f"Best Test Accuracy: {max(test_accuracies):.2f}%")

    # Check if model learned successfully
    if test_accuracies[-1] > 90.0:
        logger.info("🎉 SUCCESS! Model achieved >90% test accuracy!")
    elif test_accuracies[-1] > 80.0:
        logger.info("📈 GOOD! Model achieved >80% test accuracy!")
    else:
        logger.info("📉 Model may need more training or parameter tuning.")

    # Generate visualizations
    if config.visualizations.enabled:
        logger.info("=" * 60)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 60)

        try:
            if config.visualizations.save_training_curves:
                logger.info("Generating training curves...")
                plot_training_curves(train_losses, train_accuracies, test_losses, test_accuracies, str(output_dir))

            if config.visualizations.save_sample_predictions:
                logger.info("Generating sample predictions...")
                plot_sample_predictions(model, test_loader, config.visualizations.num_prediction_samples, str(output_dir))

            if config.visualizations.save_feature_maps:
                logger.info("Generating feature maps visualization...")
                plot_feature_maps(model, test_loader, str(output_dir))

            logger.info("✅ All visualizations generated successfully!")

        except Exception as e:
            logger.error(f"❌ Error generating visualizations: {e}")

    # Performance summary
    logger.info("=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)

    subset_info = f"subset ({config.data.subset.ratio:.1%})" if config.data.subset.enabled else "full"
    logger.info(f"📊 Dataset: MNIST ({subset_info})")
    logger.info(f"🏗️  Architecture: LeNet-5 ({trainable_params:,} parameters)")
    logger.info(f"⚡ Training Time: {total_training_time:.2f} seconds ({config.training.epochs} epochs)")
    logger.info(f"🎯 Final Accuracy: {test_accuracies[-1]:.2f}%")
    logger.info(f"📈 Improvement: {test_accuracies[-1] - test_accuracies[0]:.2f}% points")

    # Learning rate and convergence analysis
    if len(test_accuracies) > 1:
        final_improvement = test_accuracies[-1] - test_accuracies[-2]
        if abs(final_improvement) < 0.5:
            logger.info("📊 Training appears to have converged")
        else:
            logger.info("📊 Training still improving - consider more epochs")

    logger.info("=" * 60)
    logger.info("LeNet-5 MNIST Training with Fabric + Hydra Complete!")
    logger.info(f"Check the {output_dir} directory for outputs and visualizations.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
