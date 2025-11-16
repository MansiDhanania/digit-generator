"""Configuration file for MNIST CVAE project."""

import torch
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "cvae_mnist.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparameters
LATENT_DIM = 20
INPUT_DIM = 28 * 28
NUM_CLASSES = 10
HIDDEN_DIM = 400

# Training configuration
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10
WEIGHT_DECAY = 1e-5

# Image configuration
IMAGE_SIZE = 28

# Streamlit
MAX_SAMPLES = 5
