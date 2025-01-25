import time

import torch.optim as optim
import torch.nn as nn
import torch

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

    initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}

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

        for name, param in model.named_parameters():
            param_change = torch.norm(param - initial_params[name]).item()
            print(f"Epoch {epoch + 1}, Layer {name}, Parameter Change: {param_change:.4f}")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)  # Save the best model

    torch.save(model.state_dict(), LAST_MODEL_PATH)  # Save the last model
    print("Training complete. Best validation accuracy: {:.4f}".format(best_val_acc))


def evaluate_model(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# Example: Train the model
# train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)