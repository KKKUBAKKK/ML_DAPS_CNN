import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from cnn import SpectrogramCNN
from dataset import create_dataloaders

# Original code
# def monte_carlo_dropout(model, input_data, forward_passes=10):
#     model.train()  # Enable dropout during inference
#     outputs = []
#
#     for _ in range(forward_passes):
#         outputs.append(model(input_data).unsqueeze(0))
#
#     outputs = torch.cat(outputs, dim=0)
#     mean_output = outputs.mean(dim=0)
#     variance_output = outputs.var(dim=0)
#
#     return mean_output, variance_output

# Modified code
def monte_carlo_dropout(model, data_loader, forward_passes=10, device="cpu"):
    model.train()  # Enable dropout during inference
    outputs = []

    for _ in range(forward_passes):
        print(f"Forward pass {_ + 1}/{forward_passes}")
        batch_outputs = []
        i = 0
        for input_data, _ in data_loader:
            i += 1
            input_data = input_data.to(device)
            print(f"Batch size: {input_data.size(0)}, Batch {i}/{len(data_loader)}")
            with torch.no_grad():  # Disable gradient calculation
                batch_outputs.append(model(input_data).unsqueeze(0))
            torch.mps.empty_cache()
        batch_outputs = torch.cat(batch_outputs, dim=1)
        outputs.append(batch_outputs)

    outputs = torch.cat(outputs, dim=0)
    mean_output = outputs.mean(dim=0)
    variance_output = outputs.var(dim=0)

    return mean_output, variance_output


def visualize_model_uncertainty(mean, variance):
    """
    Create multiple plots to visualize model uncertainty.

    Args:
        mean (torch.Tensor): Mean predictions
        variance (torch.Tensor): Variance of predictions
    """
    plt.figure(figsize=(15, 5))

    # 1. Histogram of Variances
    plt.subplot(131)
    plt.hist(variance.numpy(), bins=30, edgecolor='black')
    plt.title('Distribution of Prediction Variances')
    plt.xlabel('Variance')
    plt.ylabel('Frequency')

    # 2. Boxplot of Variances
    # plt.subplot(132)
    # sns.boxplot(x=variance.numpy())
    # plt.title('Boxplot of Prediction Variances')
    # plt.xlabel('Variance')

    # 3. Scatter plot of Mean vs Variance
    plt.subplot(133)
    plt.scatter(mean.numpy(), variance.numpy(), alpha=0.6)
    plt.title('Mean Predictions vs Variance')
    plt.xlabel('Mean Predictions')
    plt.ylabel('Variance')

    plt.tight_layout()
    plt.savefig('model_uncertainty_plots.png')
    plt.close()


if __name__ == "__main__":
    # Check for MPS device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("MPS device not found, using CPU")

    # Prepare data loaders (from previous steps)
    print("Preparing data loaders...")
    data_dir = Path("./data/spectrograms")
    batch_size = 4
    train_loader, val_loader, test_loader = create_dataloaders(data_dir, batch_size)
    print(f"Data loaders created with batch size {batch_size}")

    # Initialize model
    # Load the best saved model before testing
    print("Loading best model...")
    model_save_path = "best_cnn.pth"
    model = SpectrogramCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_save_path, weights_only=True, map_location=device))
    print(f"Best model loaded from {model_save_path}")

    # Get uncertainty estimates with Monte Carlo Dropout
    print("Starting Monte Carlo Dropout evaluation...")
    mean, variance = monte_carlo_dropout(model, test_loader, forward_passes=2, device=device)
    print("Monte Carlo Dropout evaluation complete.")

    # print("Mean predictions: ", mean)
    # print("Variance (Uncertainty): ", variance)

    mean = mean.cpu().detach()
    variance = variance.cpu().detach()

    # Visualize uncertainty
    print("Visualizing model uncertainty...")
    visualize_model_uncertainty(mean, variance)
    print("Model uncertainty visualizations saved as model_uncertainty_plots.png")

    # Additional uncertainty metrics
    print("Uncertainty Analysis:")
    print(f"Total Variance: {variance.mean().item():.4f}")
    print(f"Max Variance: {variance.max().item():.4f}")
    print(f"Min Variance: {variance.min().item():.4f}")

    del train_loader, val_loader, test_loader, model
