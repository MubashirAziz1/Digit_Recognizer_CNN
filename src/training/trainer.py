import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from ..utils.visualization import plot_training_history

class Trainer:
    
    def __init__(self, model, train_loader, val_loader, device, 
                 criterion=None, optimizer=None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self):
 
        self.model.train()
        running_loss = 0.0
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()      # Zero the gradients
            outputs = self.model(images)    # Forward Pass
            loss = self.criterion(outputs, labels)
            loss.backward() # Backward Pass
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Gradient Clipping
            self.optimizer.step()   # Update weights
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss
    
    def validate_epoch(self):

        self.model.eval()
        val_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        
        val_loss /= len(self.val_loader.dataset)
        val_acc = correct / len(self.val_loader.dataset)
        
        return val_loss, val_acc
    
    def train(self, num_epochs: int, save_path: Optional[str] = None) -> Dict[str, List[float]]:

        best_val_acc = 0.0
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()     #Training Phase
            val_loss, val_acc = self.validate_epoch()   #Validation Phase
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch}/{num_epochs}: "
                  f"Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"Val Acc = {val_acc:.4f}")
            
            # Save best model
            if save_path and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_accuracies': self.val_accuracies
                }, save_path)
                print(f"Best model saved with validation accuracy: {val_acc:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
 
        """
        plot_training_history(
            self.train_losses, 
            self.val_losses, 
            self.val_accuracies,
            save_path=save_path
        ) 