import torch
import torch.nn as nn
from pathlib import Path
from dataset import create_dataloaders
from cnn import SpectrogramCNN
from train import train_model, evaluate_model
import copy

def reinitialize_layer(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

# Main function to train, evaluate, and save the model
def main(train, data_dir, num_epochs=25, batch_size=32, learning_rate=0.001, model_save_path="model.pth"):
    # Prepare data loaders (from previous steps)
    train_loader, val_loader, test_loader = create_dataloaders(data_dir, batch_size=batch_size)

    # For Windows
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("MPS device not found, using CPU")

    # Initialize model
    model = SpectrogramCNN(num_classes=2)

    # Train the model
    if train:
        print("Starting training...")
        train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, model_path=model_save_path)
        print("Training complete.")
    else:
        print("Skipping training...")
        model.to(device)


    # Load the best saved model before testing
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    print(f"Best model loaded from {model_save_path}")

    # Evaluate on the test set
    print("Evaluating on test set...")
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Examine reinitialization robustness
    print("\nTesting re-initialization robustness...")
    for name, module in model.named_modules():
        if hasattr(module, 'reset_parameters'):
            print(f"Reinitializing layer: {name}")

            # Create a copy of the model
            model_copy = copy.deepcopy(model)

            # Reinitialize the specific layer in the copied model
            for sub_name, sub_module in model_copy.named_modules():
                if sub_name == name:
                    sub_module.apply(reinitialize_layer)

            # Evaluate the copied model after reinitialization
            model_copy.to(device)
            reinit_test_loss, reinit_test_acc = evaluate_model(model_copy, test_loader, criterion, device)
            print(
                f"After reinitializing {name}, Test Loss: {reinit_test_loss:.4f}, Test Accuracy: {reinit_test_acc:.4f}")

# Call the main function with your dataset path
if __name__ == "__main__":
    data_dir = Path("./spectrograms")  # Update this to your dataset path
    main(True, data_dir, num_epochs=25, batch_size=32, learning_rate=0.001, model_save_path="spectrogram_cnn_best.pth")