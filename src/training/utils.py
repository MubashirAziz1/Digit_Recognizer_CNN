import torch
import os
from typing import Optional, Dict, Any

def save_checkpoint(model, optimizer, epoch, val_acc, filepath, 
                    train_losses, val_losses,val_accuracies):
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'train_losses': train_losses or [],
        'val_losses': val_losses or [],
        'val_accuracies': val_accuracies or []
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath):

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint['epoch']}, Validation Accuracy: {checkpoint['val_acc']:.4f}")
    
    return checkpoint 