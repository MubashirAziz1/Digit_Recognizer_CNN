import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


# Add src to path
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.dataset import MNISTDataset
from src.data.data_preprocessing import train_dataloader, val_dataloader, split_train_val, load_traintest_data
from src.models.cnn import CNN
from src.training.trainer import Trainer
from config import Config
from src.utils.visualization import plot_training_history, plot_confusion_matrix


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(config: Config):
    """
    Train the digit recognizer model.

    """
    print("Starting model training...")
    
    set_random_seed(config.random_seed)
    
    # Load data 
    train_images, train_labels, _ = load_traintest_data(config)
    
    #Split the train and val data
    x_train, y_train, x_val, y_val = split_train_val(train_images, train_labels, Config)

    # Create data loaders
    train_loader = train_dataloader(x_train, y_train, Config)
    val_loader = val_dataloader(x_val, y_val, Config)

    # Create model
    device = torch.device(config.device)
    model = CNN(num_classes=config.num_classes).to(device)
    print(f"Model created and moved to {device}")
    
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer
    )
    
    # Train the model
    print(f"Training for {config.num_epochs} epochs...")
    history = trainer.train(
        num_epochs=config.num_epochs,
        save_path=config.best_model_filepath
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        history['val_accuracies'],
        save_path='training_history.png'
    )
    
    # Evaluate on validation set
    print("Evaluating model...")
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
    plot_confusion_matrix(
        all_labels, 
        all_preds,
        save_path='confusion_matrix.png'
    )
    
    print("Training completed successfully!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train digit recognizer model')
    parser.add_argument('--data_path', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--model_path', type=str, default='models/saved_models',
                       help='Path to save models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        data_dir=args.data_path,
        model_dir=args.model_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        random_seed=args.seed
    )
    
    # Train model
    train_model(config)


if __name__ == '__main__':
    main() 