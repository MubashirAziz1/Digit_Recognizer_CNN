import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Data paths
    data_dir: str = "data/raw"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    
    # Model parameters
    num_classes: int = 10
    batch_size: int = 128
    num_epochs: int = 50
    learning_rate: float = 0.001
    
    # Training parameters
    validation_split: float = 0.2
    random_seed: int = 42
    
    # Model saving
    model_dir: str = "models/saved_models"
    best_model_path: str = "best_model.pth"
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def train_csv_path(self):
        """Get full path to training CSV file."""
        return os.path.join(self.data_dir, self.train_csv)
    
    @property
    def test_csv_path(self):
        """Get full path to test CSV file."""
        return os.path.join(self.data_dir, self.test_csv)
    
    @property
    def best_model_filepath(self):
        """Get full path to best model file."""
        return os.path.join(self.model_dir, self.best_model_path)
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'data_dir': self.data_dir,
            'train_csv': self.train_csv,
            'test_csv': self.test_csv,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'random_seed': self.random_seed,
            'model_dir': self.model_dir,
            'best_model_path': self.best_model_path,
            'device': self.device
        }


def get_default_config() -> Config:
    """
    Get default configuration.
    
    Returns:
        Config: Default configuration object
    """
    return Config()


def load_config_from_dict(config_dict: dict) -> Config:
    """
    Load configuration from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Config: Configuration object
    """
    return Config(**config_dict)