"""MNIST Conditional Variational Autoencoder (CVAE) Training Script.

This module implements a CVAE model for generating handwritten digits.
The model learns to generate new digit images conditioned on the digit class.
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import HIDDEN_DIM, INPUT_DIM, LATENT_DIM, NUM_CLASSES, MODEL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder for digit generation.

    Architecture:
        - Encoder: image (784) + label (10) -> hidden (400) -> latent (20)
        - Decoder: latent (20) + label (10) -> hidden (400) -> image (784)
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        input_dim: int = INPUT_DIM,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """Initialize CVAE model."""
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Encoder
        self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log-variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image and label to latent distribution parameters."""
        x = torch.cat([x, y], dim=1)
        h1 = torch.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Decode latent vector and label to reconstructed image."""
        z = torch.cat([z, y], dim=1)
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the CVAE."""
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar


def vae_loss(
    recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute CVAE loss (reconstruction + KL divergence)."""
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = bce + kld
    return total_loss, bce, kld


def train_epoch(
    model: CVAE, train_loader: DataLoader, optimizer: optim.Optimizer, device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.view(-1, INPUT_DIM).to(device)
        labels_onehot = torch.nn.functional.one_hot(labels, NUM_CLASSES).float().to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, labels_onehot)
        loss, bce, kld = vae_loss(recon_batch, data, mu, logvar)

        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            logger.info(
                f"  Batch [{batch_idx + 1}/{len(train_loader)}] - "
                f"Loss: {loss.item() / len(data):.4f}"
            )

    return total_loss / len(train_loader.dataset)


def train_model(
    model: CVAE,
    train_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train the CVAE model."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info(f"Starting training on device: {device}")
    logger.info(f"Epochs: {epochs}, Learning Rate: {learning_rate}")

    for epoch in range(epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, device)
        logger.info(f"Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}")

    logger.info("Training completed!")


def load_data(batch_size: int = 128) -> DataLoader:
    """Load MNIST training dataset."""
    logger.info("Loading MNIST dataset...")
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=".", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Dataset loaded: {len(train_dataset)} images")
    return train_loader


def save_model(model: CVAE, save_path: Path = MODEL_PATH) -> None:
    """Save model weights."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = CVAE(
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
    ).to(device)

    logger.info(f"Model initialized on device: {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load data
    train_loader = load_data(batch_size=128)

    # Train model
    train_model(model, train_loader, epochs=10, learning_rate=1e-3, device=device)

    # Save model
    save_model(model, MODEL_PATH)

    logger.info("✓ Training pipeline completed successfully!")