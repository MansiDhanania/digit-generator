import json
import os
import matplotlib.pyplot as plt

def load_json(filepath):
    """Utility function to load data from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    with open(filepath, "r") as f:
        try:
            data = json.load(f)
            if not data:  # Check if the file is empty
                raise ValueError(f"The file {filepath} is empty.")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {filepath}: {e}")

def plot_training_logs(log_file, output_dir):
    # Load training logs
    try:
        logs = load_json(log_file)
    except (FileNotFoundError, ValueError) as e:
        raise RuntimeError(f"Failed to load training logs: {e}")

    if not isinstance(logs, dict) or "train_loss" not in logs:
        raise KeyError("The training logs must contain a 'train_loss' key.")

    # Plot training loss
    plt.figure()
    plt.plot(logs["train_loss"], label="Training Loss")
    if "val_loss" in logs:
        plt.plot(logs["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

def main():
    log_file = "static/plots/training_logs.json"
    output_dir = "static/plots"
    plot_training_logs(log_file, output_dir)

if __name__ == "__main__":
    main()