import time

import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

from sklearn.metrics import f1_score


MODEL_PATH = "./model.pth"
LAST_MODEL_PATH = "./last_model.pth"

def train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=0.001, model_path=MODEL_PATH):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # For classification tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to GPU if available
    # For Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("MPS device not found, using CPU")

    # For Windows
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights

            # Track loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        print(f"Ending epoch {epoch + 1}/{num_epochs} at {time.strftime('%H:%M:%S', time.localtime())}, "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)  # Save the best model

    torch.save(model.state_dict(), LAST_MODEL_PATH)  # Save the last model
    print("Training complete. Best validation accuracy: {:.4f}".format(best_val_acc))


def evaluate_model(model, data_loader, criterion, device):
    model.train()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient computation during evaluation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect predictions and labels for F1 score calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total

    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average="weighted")  # Change "weighted" if you need macro or micro F1 score
    print(f"F1 Score: {f1:.4f}")

    return epoch_loss, epoch_acc


def mc_dropout_predictions(model, data_loader, num_samples, device):
    model.train()  # Aktywuj dropout
    all_predictions = []

    with torch.no_grad():
        print("Starting Monte Carlo Dropout evaluation...")
        for inputs, _ in data_loader:
            print(f"Batch {len(all_predictions) + 1}/{len(data_loader)}")
            inputs = inputs.to(device)

            # Wykonaj wielokrotne predykcje z aktywnym dropoutem
            predictions = []
            for _ in range(num_samples):
                print(f"Sample {_ + 1}/{num_samples}")
                outputs = torch.softmax(model(inputs), dim=1)
                predictions.append(outputs.cpu().numpy())

            predictions = np.array(predictions)  # shape: (num_samples, batch_size, num_classes)
            all_predictions.append(predictions)

    print("Monte Carlo Dropout evaluation complete.")
    return all_predictions  # Zwraca predykcje dla każdego przykładu


def train_ensemble(models, train_loader, val_loader, num_epochs=25, learning_rate=0.001):
    for model in models:
        train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate)

def ensemble_predictions(models, inputs):
    outputs = []
    with torch.no_grad():
        for model in models:
            outputs.append(model(inputs))
    return torch.stack(outputs)
# Example: Train the model
# train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)


import matplotlib.pyplot as plt

def plot_uncertainty(mean, uncertainty):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(mean)), mean, yerr=uncertainty, alpha=0.7, capsize=4)
    plt.xlabel('Class')
    plt.ylabel('Prediction probability')
    plt.title('Uncertainty estimation with Monte Carlo Dropout')
    plt.show()

# plot_uncertainty(predictions_mean[0], predictions_uncertainty[0])  # Przykład dla jednego elementu