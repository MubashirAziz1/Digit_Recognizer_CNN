#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import Config
from src.data import MNISTDataset, get_test_transforms, val_dataloader, split_train_val, load_traintest_data
from src.models import create_cnn_model
from src.utils import plot_confusion_matrix, plot_model_predictions


def load_model(model_path, device):
    """
    Load a trained model from checkpoint.
    
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model
    model = create_cnn_model(device=device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {model_path}")
    print(f"Training epoch: {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    
    return model, checkpoint


def evaluate_model(model: torch.nn.Module, val_loader: DataLoader, 
                  device: torch.device) -> tuple:
    """
    Evaluate model on validation data.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run evaluation on
        
    Returns:
        tuple: (predictions, true_labels, probabilities)
    """
    print("Evaluating model...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1} batches...")
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    print(f"Evaluation completed. Shape: {all_preds.shape}")
    
    return all_preds, all_labels, all_probs


def print_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray):
    """
    Print comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = np.diag(cm) / np.sum(cm, axis=1)
    
    print("\nPer-class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"  Class {i}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # Confusion matrix summary
    print("\nConfusion Matrix Summary:")
    print(f"  Total samples: {len(y_true)}")
    print(f"  Correct predictions: {np.sum(y_true == y_pred)}")
    print(f"  Incorrect predictions: {np.sum(y_true != y_pred)}")
    
    # Confidence analysis
    max_probs = np.max(y_probs, axis=1)
    print(f"\nConfidence Analysis:")
    print(f"  Average confidence: {np.mean(max_probs):.4f}")
    print(f"  Min confidence: {np.min(max_probs):.4f}")
    print(f"  Max confidence: {np.max(max_probs):.4f}")
    
    # Find most confident and least confident predictions
    correct_mask = y_true == y_pred
    if np.any(correct_mask) and np.any(~correct_mask):
        correct_conf = max_probs[correct_mask]
        incorrect_conf = max_probs[~correct_mask]
        print(f"  Average confidence (correct): {np.mean(correct_conf):.4f}")
        print(f"  Average confidence (incorrect): {np.mean(incorrect_conf):.4f}")


def analyze_errors(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, 
                  val_images: np.ndarray, num_examples: int = 5):
    """
    Analyze prediction errors.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        val_images: Validation images
        num_examples: Number of error examples to show
    """
    print(f"\n" + "="*50)
    print("ERROR ANALYSIS")
    print("="*50)
    
    # Find incorrect predictions
    error_mask = y_true != y_pred
    error_indices = np.where(error_mask)[0]
    
    if len(error_indices) == 0:
        print("No errors found!")
        return
    
    print(f"Total errors: {len(error_indices)}")
    
    # Show some error examples
    num_examples = min(num_examples, len(error_indices))
    selected_errors = error_indices[:num_examples]
    
    print(f"\nShowing {num_examples} error examples:")
    for i, idx in enumerate(selected_errors):
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        confidence = y_probs[idx][pred_label]
        
        print(f"  Example {i+1}: True={true_label}, Pred={pred_label}, Confidence={confidence:.4f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate digit recognizer model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save evaluation plots')
    
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
        
        # Load validation data
        train_images, train_labels, _ = load_traintest_data()
        _, _, val_images, val_labels = split_train_val(train_images, train_labels)

        # Create validation loader
        val_loader = val_dataloader(val_images, val_labels, args.batch_size)
        
        # Evaluate model
        predictions, true_labels, probabilities = evaluate_model(model, val_loader, device)
        
        # Print evaluation metrics
        print_evaluation_metrics(true_labels, predictions, probabilities)
        
        # Analyze errors
        analyze_errors(true_labels, predictions, probabilities, val_images)
        
        # Create plots
        if args.save_plots:
            print("\nCreating evaluation plots...")
            plot_confusion_matrix(
                true_labels, 
                predictions,
                save_path='evaluation_confusion_matrix.png'
            )
            
            # Note: plot_model_predictions would need the actual images from the loader
            # This is a simplified version
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 