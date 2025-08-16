# Digit Recognizer

A deep learning project for handwritten digit recognition using Convolutional Neural Networks (CNN) with PyTorch.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a CNN-based digit recognizer trained on the MNIST dataset. The model achieves high accuracy in recognizing handwritten digits (0-9) and can be used for various applications like OCR systems, form processing, and educational tools.

## ✨ Features

- **Convolutional Neural Network**: Modern CNN architecture with batch normalization and dropout
- **Data Augmentation**: Random rotation for improved generalization
- **Training Pipeline**: Complete training loop with validation and early stopping
- **Model Evaluation**: Confusion matrix and per-class accuracy analysis
- **Prediction Pipeline**: Ready-to-use prediction functionality
- **Modular Design**: Clean, maintainable code structure

## 📁 Project Structure

```
digit-recognizer/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── utils.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── visualization.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── saved_models/
├── notebooks/
│   └── exploration.ipynb
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── tests/
│   └── __init__.py
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/digit-recognizer.git
   cd digit-recognizer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Training the Model

```bash
python scripts/train.py --data_path data/raw --model_path models/saved_models --epochs 50
```

### Making Predictions

```bash
python scripts/predict.py --model_path models/saved_models/best_model.pth --input_path data/test.csv
```

### Evaluating the Model

```bash
python scripts/evaluate.py --model_path models/saved_models/best_model.pth --data_path data/raw
```

## 🏗️ Model Architecture

The CNN architecture consists of:

- **Feature Extraction**: 3 convolutional layers with batch normalization and ReLU activation
- **Pooling**: MaxPool2d layers for dimensionality reduction
- **Classification**: Fully connected layers with dropout for regularization
- **Output**: 10 classes (digits 0-9)

## 📊 Results

- **Validation Accuracy**: ~99.4%
- **Training Time**: ~50 epochs
- **Model Size**: Optimized for inference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MNIST dataset creators
- PyTorch community
- Kaggle for hosting the competition 