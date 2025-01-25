import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SpectrogramCNN, self).__init__()

        # Input is grayscale (1 channel), so input channels = 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Based on 224x224 input size after 3 pooling layers
        self.fc2 = nn.Linear(512, num_classes)  # Output layer (for binary classification, num_classes=2)

        # Dropout (optional, helps prevent overfitting)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with ReLU activation and Max Pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: (32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (64, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # Output: (128, 28, 28)

        # Flatten the tensor for fully connected layers
        x = x.view(-1, 128 * 28 * 28)  # Flattening the output of conv layers

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output: (batch_size, num_classes)

        return x


    def enable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


def monte_carlo_dropout(model, inputs, num_samples=50):
    model.enable_dropout()
    model.eval()
    outputs = []
    with torch.no_grad():
        for _ in range(num_samples):
            outputs.append(model(inputs))
    return torch.stack(outputs)

# Example: Initialize the CNN
# model = SpectrogramCNN(num_classes=2)
# print(model)