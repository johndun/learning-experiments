#!/usr/bin/env python3
"""
LeNet MNIST Training Script

A complete implementation of LeNet-5 architecture for MNIST digit classification
using PyTorch. Optimized for CPU training with rapid feedback and visualization.

This script demonstrates:
- LeNet-5 convolutional neural network architecture
- MNIST dataset loading and preprocessing
- Training loop with progress tracking
- Model evaluation and accuracy computation
- Training visualization and sample predictions
- CPU-optimized hyperparameters for fast iteration

Author: Learning Experiments
"""

import os
import time
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")


class LeNet5(nn.Module):
    """
    LeNet-5 Convolutional Neural Network for MNIST classification.

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

    def __init__(self):
        super(LeNet5, self).__init__()

        # Feature extraction layers (convolutional)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification layers (fully connected)
        # After conv layers: 28->24->12->8->4, so final feature map is 16*4*4=256
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
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


def load_mnist_data(batch_size: int = 64, use_subset: bool = True, subset_ratio: float = 0.1) -> Tuple[DataLoader, DataLoader]:
    """
    Load and preprocess MNIST dataset.

    Args:
        batch_size: Batch size for data loaders
        use_subset: Whether to use only a subset of data for faster training
        subset_ratio: Fraction of data to use if use_subset is True

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),                    # Convert PIL Image to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Download and load datasets
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Use subset for faster training/testing
    if use_subset:
        train_size = int(len(train_dataset) * subset_ratio)
        test_size = int(len(test_dataset) * subset_ratio)

        # Create random subsets
        train_indices = torch.randperm(len(train_dataset))[:train_size]
        test_indices = torch.randperm(len(test_dataset))[:test_size]

        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

        print(f"Using subset: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    else:
        print(f"Using full dataset: {len(train_dataset)} training samples, {len(test_dataset)} test samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Single-threaded for CPU optimization
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device, epoch: int) -> Tuple[float, float]:
    """
    Train model for one epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Device to run training on
        epoch: Current epoch number

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch + 1} Training:")
    print("-" * 50)

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            progress = 100.0 * batch_idx / len(train_loader)
            current_acc = 100.0 * correct / total
            print(f"Batch {batch_idx:3d}/{len(train_loader)} ({progress:5.1f}%) | "
                  f"Loss: {loss.item():.6f} | Acc: {current_acc:.2f}%")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    print(f"Training Results: Loss={avg_loss:.6f}, Accuracy={accuracy:.2f}%")
    return avg_loss, accuracy


def test_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module,
               device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on test dataset.

    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to run evaluation on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Calculate loss
            total_loss += criterion(output, target).item()

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    print(f"Test Results: Loss={avg_loss:.6f}, Accuracy={accuracy:.2f}%")
    return avg_loss, accuracy


def plot_training_curves(train_losses: List[float], train_accuracies: List[float],
                        test_losses: List[float], test_accuracies: List[float],
                        save_dir: str = "outputs"):
    """
    Plot training and validation curves.

    Args:
        train_losses: List of training losses per epoch
        train_accuracies: List of training accuracies per epoch
        test_losses: List of test losses per epoch
        test_accuracies: List of test accuracies per epoch
        save_dir: Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping training curves plot.")
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
    loss_path = os.path.join(save_dir, "lenet_training_curves.png")
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {loss_path}")
    plt.close()


def plot_sample_predictions(model: nn.Module, test_loader: DataLoader, device: torch.device,
                           num_samples: int = 16, save_dir: str = "outputs"):
    """
    Plot sample predictions from the model.

    Args:
        model: Trained neural network model
        test_loader: Test data loader
        device: Device to run inference on
        num_samples: Number of samples to display
        save_dir: Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping sample predictions plot.")
        return

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Get one batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

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
    fig.suptitle('LeNet-5 Sample Predictions on MNIST', fontsize=16, fontweight='bold')

    for i in range(num_samples):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        # Convert tensor to numpy for plotting
        image = images[i].cpu().squeeze().numpy()
        true_label = labels[i].cpu().item()
        pred_label = predictions[i].cpu().item()

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
    pred_path = os.path.join(save_dir, "lenet_sample_predictions.png")
    plt.savefig(pred_path, dpi=300, bbox_inches='tight')
    print(f"Sample predictions saved to: {pred_path}")
    plt.close()


def plot_feature_maps(model: nn.Module, test_loader: DataLoader, device: torch.device,
                     save_dir: str = "outputs"):
    """
    Visualize learned feature maps from convolutional layers.

    Args:
        model: Trained neural network model
        test_loader: Test data loader
        device: Device to run inference on
        save_dir: Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping feature maps plot.")
        return

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Get one sample
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    sample_image = images[0:1].to(device)  # Take first image, keep batch dimension
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

    plt.suptitle('LeNet-5 Learned Feature Maps', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save plot
    features_path = os.path.join(save_dir, "lenet_feature_maps.png")
    plt.savefig(features_path, dpi=300, bbox_inches='tight')
    print(f"Feature maps saved to: {features_path}")
    plt.close()


def train_model_version(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                       optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device,
                       config: dict, version_name: str) -> Tuple[List[float], List[float], List[float], List[float], float]:
    """
    Train a model version and return training metrics.

    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Device to run training on
        config: Training configuration dictionary
        version_name: Name for this version (e.g., "Standard" or "Compiled")

    Returns:
        Tuple of (train_losses, train_accuracies, test_losses, test_accuracies, total_time)
    """
    print(f"\n" + "=" * 60)
    print(f"TRAINING {version_name.upper()} VERSION")
    print("=" * 60)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    training_start_time = time.time()

    for epoch in range(config['epochs']):
        epoch_start_time = time.time()

        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Evaluate on test set
        print("Evaluating on test set...")
        test_loss, test_acc = test_model(model, test_loader, criterion, device)

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
        print("=" * 50)

    total_training_time = time.time() - training_start_time

    return train_losses, train_accuracies, test_losses, test_accuracies, total_training_time


def main():
    """Main function to demonstrate LeNet-5 training on MNIST with torch.compile speedup comparison."""
    print("LeNet-5 MNIST Training - torch.compile Speedup Demo")
    print("=" * 55)

    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print("CUDA available but using CPU-optimized settings")
        device = torch.device("cpu")  # Force CPU for consistent results

    # Set random seeds for reproducibility
    torch.manual_seed(42)

    # Hyperparameters (CPU optimized for rapid feedback)
    config = {
        'batch_size': 64,           # Good balance for CPU
        'learning_rate': 0.01,      # Higher LR for faster convergence
        'momentum': 0.9,            # SGD momentum
        'epochs': 5,                # Few epochs for rapid feedback
        'use_subset': True,         # Use subset for ultra-fast training
        'subset_ratio': 0.1,        # 10% of data for rapid iteration
        'compare_compile': True,    # Enable torch.compile comparison
    }

    print("\nTraining Configuration:")
    print("-" * 30)
    for key, value in config.items():
        print(f"{key:15}: {value}")
    print()

    # Load data
    start_time = time.time()
    train_loader, test_loader = load_mnist_data(
        batch_size=config['batch_size'],
        use_subset=config['use_subset'],
        subset_ratio=config['subset_ratio']
    )
    data_time = time.time() - start_time
    print(f"Data loading completed in {data_time:.2f} seconds")

    # Create model
    print("\nInitializing LeNet-5 model...")
    model = LeNet5().to(device)

    # Print model architecture
    print("\nModel Architecture:")
    print("-" * 30)
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Define loss function and optimizer
    criterion = nn.NLLLoss()  # Negative Log Likelihood for log_softmax output
    optimizer = optim.SGD(model.parameters(),
                         lr=config['learning_rate'],
                         momentum=config['momentum'])

    print(f"\nOptimizer: SGD(lr={config['learning_rate']}, momentum={config['momentum']})")
    print(f"Loss function: {criterion}")

    # torch.compile comparison training
    if config['compare_compile']:
        print("\n" + "=" * 70)
        print("TORCH.COMPILE SPEEDUP COMPARISON")
        print("=" * 70)
        print("Training identical models with and without torch.compile optimization")
        print("This comparison will show potential performance improvements from compilation.")
        print()

        # ===== STANDARD MODEL TRAINING =====
        # Reset model and optimizer for fair comparison
        torch.manual_seed(42)  # Reset seed for reproducible weights
        model_standard = LeNet5().to(device)
        optimizer_standard = optim.SGD(model_standard.parameters(),
                                     lr=config['learning_rate'],
                                     momentum=config['momentum'])

        # Train standard model
        (train_losses_std, train_accuracies_std,
         test_losses_std, test_accuracies_std,
         training_time_std) = train_model_version(
            model_standard, train_loader, test_loader,
            optimizer_standard, criterion, device, config, "Standard"
        )

        # ===== COMPILED MODEL TRAINING =====
        # Reset model and optimizer for fair comparison
        torch.manual_seed(42)  # Reset seed for identical weights
        model_compiled = LeNet5().to(device)

        # Apply torch.compile optimization
        print(f"\nApplying torch.compile optimization...")
        model_compiled = torch.compile(model_compiled)
        print("✅ Model compiled successfully!")

        optimizer_compiled = optim.SGD(model_compiled.parameters(),
                                     lr=config['learning_rate'],
                                     momentum=config['momentum'])

        # Train compiled model
        (train_losses_comp, train_accuracies_comp,
         test_losses_comp, test_accuracies_comp,
         training_time_comp) = train_model_version(
            model_compiled, train_loader, test_loader,
            optimizer_compiled, criterion, device, config, "Compiled"
        )

        # ===== PERFORMANCE COMPARISON =====
        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON RESULTS")
        print("=" * 70)

        # Calculate speedup metrics
        speedup_ratio = training_time_std / training_time_comp
        time_saved = training_time_std - training_time_comp
        speedup_percent = ((training_time_std - training_time_comp) / training_time_std) * 100

        # Training time comparison
        print(f"\n📊 TRAINING TIME COMPARISON:")
        print(f"{'Standard Model:':<20} {training_time_std:>8.2f} seconds")
        print(f"{'Compiled Model:':<20} {training_time_comp:>8.2f} seconds")
        print(f"{'Time Saved:':<20} {time_saved:>8.2f} seconds")
        print(f"{'Speedup Ratio:':<20} {speedup_ratio:>8.2f}x")
        print(f"{'Speedup Percent:':<20} {speedup_percent:>7.1f}%")

        # Accuracy comparison
        print(f"\n🎯 ACCURACY COMPARISON:")
        print(f"{'Standard Final:':<20} {test_accuracies_std[-1]:>7.2f}%")
        print(f"{'Compiled Final:':<20} {test_accuracies_comp[-1]:>7.2f}%")
        accuracy_diff = test_accuracies_comp[-1] - test_accuracies_std[-1]
        print(f"{'Accuracy Diff:':<20} {accuracy_diff:>+7.2f}%")

        # Performance analysis
        print(f"\n⚡ SPEEDUP ANALYSIS:")
        if speedup_percent > 10:
            print(f"🚀 EXCELLENT! torch.compile provided {speedup_percent:.1f}% speedup")
            print("   Compilation optimization significantly improved training performance!")
        elif speedup_percent > 5:
            print(f"📈 GOOD! torch.compile provided {speedup_percent:.1f}% speedup")
            print("   Compilation optimization provided measurable performance gain.")
        elif speedup_percent > 0:
            print(f"📊 MODEST! torch.compile provided {speedup_percent:.1f}% speedup")
            print("   Compilation optimization provided small performance gain.")
        else:
            print(f"📉 SLOWER! torch.compile provided {speedup_percent:.1f}% change")
            print("   Compilation overhead outweighed performance benefits.")
            print("   This is common for small models, CPU training, or short training runs.")

        # Compilation overhead analysis
        if training_time_comp > training_time_std:
            print(f"\n📋 COMPILATION OVERHEAD ANALYSIS:")
            print("   torch.compile introduces overhead during the first epoch(s) for graph compilation.")
            print("   Benefits typically appear in longer training runs or more complex models.")
            print("   Consider torch.compile for: GPU training, large models, or many epochs.")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if speedup_percent > 5:
            print("✅ torch.compile is beneficial for this workload - consider using it in production")
        else:
            print("⚠️  torch.compile benefits are minimal - evaluate based on specific use case")

        print("📝 Note: Speedup can vary based on model complexity, hardware, and PyTorch version")

        # Store results for visualization (use compiled model results)
        train_losses = train_losses_comp
        train_accuracies = train_accuracies_comp
        test_losses = test_losses_comp
        test_accuracies = test_accuracies_comp
        total_training_time = training_time_comp
        model = model_compiled

    else:
        # Standard training without comparison
        print("\n" + "=" * 60)
        print("STARTING STANDARD TRAINING")
        print("=" * 60)

        (train_losses, train_accuracies,
         test_losses, test_accuracies,
         total_training_time) = train_model_version(
            model, train_loader, test_loader,
            optimizer, criterion, device, config, "Standard"
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Final training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {total_training_time/config['epochs']:.2f} seconds")

    # Final results
    print("\nFinal Training Results:")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Best Test Accuracy: {max(test_accuracies):.2f}%")

    # Check if model learned successfully
    if test_accuracies[-1] > 90.0:
        print("🎉 SUCCESS! Model achieved >90% test accuracy!")
    elif test_accuracies[-1] > 80.0:
        print("📈 GOOD! Model achieved >80% test accuracy!")
    else:
        print("📉 Model may need more training or parameter tuning.")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    try:
        # Training curves
        print("Generating training curves...")
        plot_training_curves(train_losses, train_accuracies, test_losses, test_accuracies)

        # Sample predictions
        print("Generating sample predictions...")
        plot_sample_predictions(model, test_loader, device)

        # Feature maps
        print("Generating feature maps visualization...")
        plot_feature_maps(model, test_loader, device)

        print("✅ All visualizations generated successfully!")

    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")

    # Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    print(f"📊 Dataset: MNIST ({'subset' if config['use_subset'] else 'full'})")
    print(f"🏗️  Architecture: LeNet-5 ({trainable_params:,} parameters)")
    print(f"⚡ Training Time: {total_training_time:.2f} seconds ({config['epochs']} epochs)")
    print(f"🎯 Final Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"📈 Improvement: {test_accuracies[-1] - test_accuracies[0]:.2f}% points")

    # Learning rate and convergence analysis
    if len(test_accuracies) > 1:
        final_improvement = test_accuracies[-1] - test_accuracies[-2]
        if abs(final_improvement) < 0.5:
            print("📊 Training appears to have converged")
        else:
            print("📊 Training still improving - consider more epochs")

    print("\n" + "=" * 60)
    print("LeNet-5 MNIST Training Complete!")
    print("Check the outputs/ directory for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
