# MNIST Digit Generator

A **Conditional Variational Autoencoder (CVAE)** that generates realistic handwritten digits. This project demonstrates deep generative modeling with a user-friendly web interface.

## Project Overview

This project highlights:
- **Generative AI**: Training a CVAE to learn the latent distribution of handwritten digits
- **PyTorch**: Modern deep learning implementation with proper architecture design
- **Web App**: Interactive Streamlit interface for real-time digit generation

## Features

- **Conditional Generation**: Generate any digit (0-9) on demand
- **Latent Space Exploration**: Sample from the learned latent distribution
- **Interactive UI**: Web-based interface with Streamlit
- **GPU Support**: Automatic GPU acceleration when available
- **Model Persistence**: Pre-trained weights included for immediate use

## Model Architecture

### Conditional Variational Autoencoder (CVAE)

**Encoder** (Inference Network):
- Input: Flattened image (784) + one-hot label (10) → 794 dims
- Hidden layer: 400 neurons with ReLU activation
- Output: Mean and log-variance vectors (20 dims each)

**Latent Space**:
- Dimension: 20
- Distribution: Gaussian (learned by KL divergence regularization)

**Decoder** (Generative Network):
- Input: Latent vector (20) + one-hot label (10) → 30 dims
- Hidden layer: 400 neurons with ReLU activation
- Output: Reconstructed image (784) with sigmoid activation

**Loss Function**:
- Reconstruction Loss: Binary Cross-Entropy (BCE)
- KL Divergence: Regularizes latent space to standard Gaussian
- Combined: `Loss = BCE + KLD`

### Training Details
- **Dataset**: MNIST (60000 training images)
- **Batch Size**: 128
- **Learning Rate**: 1e-3 (Adam optimizer)
- **Epochs**: 10
- **Device**: GPU if available, else CPU

## Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MansiDhanania/digit-generator.git
   cd digit-generator
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Launch the Web App
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

#### Train a New Model (Optional)
To retrain the model on MNIST:
```bash
python cvae_mnist.py
```
This will download MNIST data and save the trained model as `cvae_mnist.pth`.

## How It Works

1. **Model Training** (`cvae_mnist.py`):
   - Loads MNIST dataset
   - Trains CVAE to encode images into a latent space
   - Learns to generate new images conditioned on digit labels
   - Saves model weights to `cvae_mnist.pth`

2. **Web Interface** (`app.py`):
   - Loads pre-trained model
   - User selects a digit (0-9)
   - Generates 5 random samples from the learned distribution
   - Displays generated images in real-time

## Project Structure

```
digit-generator/
├── app.py                 # Streamlit web application
├── cvae_mnist.py          # Model training script
├── cvae_mnist.pth         # Pre-trained model weights (~2MB)
├── config.py              # Configuration and hyperparameters
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .gitignore             # Git ignore rules
```

## Configuration

Edit `config.py` to customize:
- Model hyperparameters (latent dimension, batch size, learning rate)
- Training parameters (epochs, device selection)
- File paths and model checkpoint location

## Performance

- **Inference Time**: ~10ms per 5 images (GPU), ~50ms (CPU)
- **Model Size**: ~2MB
- **Memory**: ~200MB during training, ~50MB for inference

## Technical Highlights

### Variational Autoencoder (VAE) Basics
- **Encoder**: Maps input images to latent distribution
- **Latent Space**: Continuous representation learned via KL divergence
- **Decoder**: Generates new images from latent samples
- **Conditional**: Conditioned on digit class for controlled generation

### Why CVAE?
- Enables controlled generation of specific digits
- Learns smooth latent space for interpolation
- Combines supervised learning (class conditioning) with unsupervised (generative) learning

---

**Author**: Mansi Dhanania  
**Repository**: [digit-generator](https://github.com/MansiDhanania/digit-generator)
