import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class MNISTDataset(Dataset):
    """
    Custom dataset class for MNIST digit recognition.

    """
    
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        """
        img = self.images[idx]
        img = transforms.ToPILImage()(img)
        
        if self.transform:
            img = self.transform(img)
            
        if self.labels is not None:
            return img, int(self.labels[idx])
        else:
            return img 