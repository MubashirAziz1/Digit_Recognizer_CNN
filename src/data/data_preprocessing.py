import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from config import Config
from sklearn.model_selection import train_test_split
from src.data.dataset import MNISTDataset
from src.data.transforms import get_test_transforms, get_train_transforms

def load_traintest_data(config: Config):

    print("Loading data...")
    
    # Load CSV files
    train_df = pd.read_csv(config.train_csv_path)
    test_df = pd.read_csv(config.test_csv_path)
    
    # Extract labels and images
    train_labels = train_df['label'].values
    train_images = train_df.drop('label', axis=1).values
    test_images = test_df.values
    
    # Normalize pixel values
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    # Reshape to image format (28x28)
    train_images = train_images.reshape(-1, 28, 28)
    test_images = test_images.reshape(-1, 28, 28)
    
    print(f"Train images: {train_images.shape}, Train labels: {train_labels.shape}")
    print(f"Test images: {test_images.shape}")
    
    return train_images, train_labels, test_images

def split_train_val(train_images, train_labels, config: Config):

    # Split data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(
        train_images, train_labels, 
        test_size=config.validation_split, 
        stratify=train_labels,
        random_state=config.random_seed
    )

    return x_train, y_train, x_val, y_val

def train_dataloader(x_train, y_train, config: Config):

    # Get transforms
    train_transform = get_train_transforms()

    # Create datasets
    train_dataset = MNISTDataset(x_train, y_train, transform=train_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=2
    )

    return train_loader

def val_dataloader(x_val, y_val, config:Config):

    test_transform = get_test_transforms()
    val_dataset = MNISTDataset(x_val, y_val, transform=test_transform)

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=2
    )

    return val_loader

def test_dataloader(test_images, batch_size: int = 128):

    test_transform = get_test_transforms()
    test_dataset = MNISTDataset(test_images, labels = None, transform = test_transform)
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    return test_loader



