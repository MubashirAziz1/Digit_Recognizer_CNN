"""
Tests for the models module.
"""

import pytest
import torch
import numpy as np

# Add src to path for testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models import CNN


class TestCNN:
    """Test cases for the CNN model."""
    
    def test_cnn_initialization(self):
        """Test CNN model initialization."""
        model = CNN()
        assert isinstance(model, torch.nn.Module)
        assert model.num_classes == 10
    
    def test_cnn_forward_pass(self):
        """Test CNN forward pass."""
        model = CNN()
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
    
    def test_cnn_with_custom_classes(self):
        """Test CNN with custom number of classes."""
        num_classes = 5
        model = CNN(num_classes=num_classes)
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        output = model(input_tensor)
        
        assert output.shape == (batch_size, num_classes)
    
    def test_cnn_device_movement(self):
        """Test CNN model device movement."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = CNN().to(device)
            
            batch_size = 2
            input_tensor = torch.randn(batch_size, 1, 28, 28).to(device)
            output = model(input_tensor)
            
            assert output.device == device
            assert output.shape == (batch_size, 10)


class TestCreateCNNModel:
    """Test cases for the create_cnn_model function."""
    
    def test_create_cnn_model_default(self):
        """Test create_cnn_model with default parameters."""
        model = create_cnn_model()
        assert isinstance(model, CNN)
        assert model.num_classes == 10
    
    def test_create_cnn_model_custom_classes(self):
        """Test create_cnn_model with custom number of classes."""
        num_classes = 7
        model = create_cnn_model(num_classes=num_classes)
        assert model.num_classes == num_classes
    
    def test_create_cnn_model_with_device(self):
        """Test create_cnn_model with device specification."""
        device = torch.device('cpu')
        model = create_cnn_model(device=device)
        assert next(model.parameters()).device == device


class TestModelArchitecture:
    """Test cases for model architecture details."""
    
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        model = CNN()
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0
    
    def test_model_layers(self):
        """Test that model has expected layers."""
        model = CNN()
        
        # Check for convolutional layers
        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        assert len(conv_layers) == 3
        
        # Check for batch normalization layers
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert len(bn_layers) == 3
        
        # Check for linear layers
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 3
    
    def test_model_output_range(self):
        """Test that model outputs are reasonable."""
        model = CNN()
        batch_size = 8
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        
        output = model(input_tensor)
        
        # Check that outputs are finite
        assert torch.isfinite(output).all()
        
        # Check that outputs are not all the same
        assert not torch.allclose(output[0], output[1])


if __name__ == '__main__':
    pytest.main([__file__]) 