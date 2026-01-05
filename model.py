import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=128, out_features=10)
    
    def forward(self, x):
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    model = MyAwesomeModel()
    dummy_input = torch.randn(1, 1, 28, 28)
    print(f"Output shape: {model(dummy_input).shape}")