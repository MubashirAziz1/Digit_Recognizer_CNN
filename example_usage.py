#!/usr/bin/env python3
"""
Example usage of the digit recognizer modular components.

This script demonstrates how to use the different modules together
to train a model and make predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('src')

from src.data import MNISTDataset, get_train_transforms, get_test_transforms
from src.models import CNN
from src.training import Trainer
from config import Config
from src.utils import plot_training_history, plot_confusion_matrix


def create_sample_data():
    """
    Create sample data for demonstration purposes.
    In a real scenario, you would load actual MNIST data.
    """
    print("Creating sample data for demonstration...")
    
    # Create synthetic data (28x28 images)
    num_samples = 1000
    train_images = np.random.rand(num_samples, 28, 28).astype(np.float32)
    train_labels = np.random.randint(0, 10, num_samples)
    
    # Create synthetic test data
    test_images = np.random.rand(200, 28, 28).astype(np.float32)
    
    return train_images, train_labels, test_images


def main():
    """Main demonstration function."""
    print("="*60)
    print("DIGIT RECOGNIZER - MODULAR STRUCTURE DEMONSTRATION")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create sample data
    train_images, train_labels, test_images = create_sample_data()
    
    # Split data
    x_train, x_val, y_train, y_val = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(test_images)}")
    
    # Create datasets and data loaders
    print("\n1. Creating datasets and data loaders...")
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    
    train_dataset = MNISTDataset(x_train, y_train, transform=train_transform)
    val_dataset = MNISTDataset(x_val, y_val, transform=test_transform)
    test_dataset = MNISTDataset(test_images, labels=None, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("\n2. Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(num_classes=10).to(device)
    print(f"Model created on device: {device}")
    
    # Create trainer
    print("\n3. Setting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Train model (short training for demonstration)
    print("\n4. Training model...")
    print("Note: This is a demonstration with synthetic data and few epochs")
    history = trainer.train(
        num_epochs=5,  # Short training for demonstration
        save_path='models/saved_models/demo_model.pth'
    )
    
    # Plot training history
    print("\n5. Plotting training history...")
    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        history['val_accuracies']
    )
    
    # Evaluate model
    print("\n6. Evaluating model...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds)
    
    # Make predictions on test data
    print("\n7. Making predictions on test data...")
    model.eval()
    test_predictions = []
    
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            test_predictions.extend(preds.cpu().numpy())
    
    test_predictions = np.array(test_predictions)
    print(f"Test predictions shape: {test_predictions.shape}")
    print(f"Unique predicted classes: {np.unique(test_predictions)}")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nThis demonstrates the modular structure of the digit recognizer project.")
    print("In a real scenario, you would:")
    print("1. Use actual MNIST data from CSV files")
    print("2. Train for more epochs (e.g., 50)")
    print("3. Use larger batch sizes and more data")
    print("4. Save and load models for production use")
    print("\nTo use with real data, run:")
    print("python scripts/train.py --data_path data --model_path models/saved_models --epochs 50")


if __name__ == '__main__':
    main() 