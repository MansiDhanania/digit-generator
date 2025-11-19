import json
import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cvae_mnist import CVAE, train_model, load_data
from config import LATENT_DIM, INPUT_DIM, NUM_CLASSES, HIDDEN_DIM, BATCH_SIZE, LEARNING_RATE, EPOCHS, DEVICE, MODEL_PATH

def save_json(data, filepath):
    """Utility function to save data as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f)

def main():
    # Load data
    train_loader = load_data(batch_size=BATCH_SIZE)

    # Initialize model
    model = CVAE(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(DEVICE)

    # Train model
    training_logs = train_model(model, train_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE, device=DEVICE)

    # Save training logs and model weights
    save_json(training_logs, "static/plots/training_logs.json")
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    main()