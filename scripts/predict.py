import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data import MNISTDataset, load_traintest_data, get_test_transforms, test_dataloader
from src.models import CNN
from config import Config


def load_model(model_path: str, device: torch.device):
    """
    Load a trained model from checkpoint.
    
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model
    model = CNN().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {model_path}")
    print(f"Training epoch: {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    
    return model, checkpoint


def make_predictions(model: torch.nn.Module, test_loader: DataLoader, 
                    device: torch.device) -> np.ndarray:
    """
    Make predictions on test data.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run inference on
        
    Returns:
        np.ndarray: Predictions
    """
    print("Making predictions...")
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1} batches...")
    
    predictions = np.array(predictions)
    print(f"Predictions shape: {predictions.shape}")
    
    return predictions



def predict_single_image(model: torch.nn.Module, image_path: str, device: torch.device):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained model
        image_path: Path to the image file
        device: Device to run inference on
        
    Returns:
        tuple: (prediction, confidence)
    """
    # This function would need to be implemented based on the image format
    # For now, it's a placeholder
    print("Single image prediction not implemented yet.")
    return None, None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Make predictions with digit recognizer model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to test CSV file')
    parser.add_argument('--output_path', type=str, default='submission.csv',
                       help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # Load model
        model, checkpoint = load_model(args.model_path, device)
        
        # Load test data
        _, _, test_images = load_traintest_data(args.input_path)
        
        # Create test loader
        test_loader = test_dataloader(test_images, args.batch_size)
        
        # Make predictions
        predictions = make_predictions(model, test_loader, device)
        
        # Create submission file
        submission_df = create_submission(predictions, args.output_path)
        
        print("Prediction completed successfully!")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 