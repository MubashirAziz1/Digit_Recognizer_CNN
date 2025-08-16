import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from typing import List, Optional, Tuple


def plot_training_history(train_losses: List[float], val_losses: List[float], 
                         val_accuracies: List[float], save_path: Optional[str] = None):

    epochs = np.arange(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy', color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot learning curves together
    plt.subplot(1, 3, 3)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    line2 = ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    line3 = ax2.plot(epochs, val_accuracies, 'g-', label='Val Accuracy', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='black')
    ax2.set_ylabel('Accuracy', color='green')
    ax1.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title('Training History')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: Optional[List[str]] = None, 
                         save_path: Optional[str] = None):
    """

    Plot confusion matrix.
    
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 5))
    
    # Plot raw confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Raw Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Plot normalized confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()
    
    # Print per-class accuracy
    per_class_acc = np.diag(cm_norm)
    print("\nPer-class accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"Class {i}: {acc:.4f}")


def plot_sample_images(images: torch.Tensor, labels: torch.Tensor, 
                      num_samples: int = 5, save_path: Optional[str] = None):
    """
    Plot sample images from the dataset.
    
    """
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        img = images[i].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')
    
    plt.suptitle('Sample Training Images')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample images plot saved to {save_path}")
    
    plt.show()


def plot_model_predictions(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                          device: torch.device, num_samples: int = 10, 
                          save_path: Optional[str] = None):
    """
    
    Plot model predictions on sample data.
    
    """
    model.eval()
    images, labels = next(iter(data_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()
        probabilities = torch.softmax(outputs, dim=1).cpu()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    for i in range(num_samples):
        # Plot image
        img = images[i].cpu().squeeze().numpy()
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'True: {labels[i]}\nPred: {predictions[i]}')
        axes[0, i].axis('off')
        
        # Plot prediction probabilities
        probs = probabilities[i].numpy()
        axes[1, i].bar(range(10), probs)
        axes[1, i].set_title(f'Confidence: {probs[predictions[i]]:.3f}')
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_xticks(range(10))
        axes[1, i].set_xticklabels(range(10), rotation=45)
    
    plt.suptitle('Model Predictions')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model predictions plot saved to {save_path}")
    
    plt.show()


def set_plot_style():
    """Set consistent plot style for all visualizations."""
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3 