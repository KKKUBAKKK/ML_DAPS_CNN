import torch
import torch.nn as nn
import torch.nn.functional as F

class ModifiedSpectrogramCNN(nn.Module):
    def __init__(self, num_classes=2, activation='relu', use_skip_connections=False):
        super(ModifiedSpectrogramCNN, self).__init__()

        self.use_skip_connections = use_skip_connections

        # Define activation function
        self.activation_fn = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'sigmoid': torch.sigmoid
        }[activation]

        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Adjusted based on additional pooling layers
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers with optional skip connections
        x1 = self.pool(self.activation_fn(self.bn1(self.conv1(x))))  # Output: (32, 112, 112)
        x2 = self.pool(self.activation_fn(self.bn2(self.conv2(x1))))  # Output: (64, 56, 56)

        if self.use_skip_connections:
            x2 = x2 + x1  # Add skip connection

        x3 = self.pool(self.activation_fn(self.bn3(self.conv3(x2))))  # Output: (128, 28, 28)
        x4 = self.pool(self.activation_fn(self.bn4(self.conv4(x3))))  # Output: (256, 14, 14)

        # Flatten
        x = x4.view(-1, 256 * 14 * 14)

        # Fully connected layers
        x = self.dropout1(self.activation_fn(self.fc1(x)))
        x = self.fc2(x)

        return x

# Example usage
if __name__ == "__main__":
    model = ModifiedSpectrogramCNN(num_classes=2, activation='leaky_relu', use_skip_connections=True)
    print(model)
