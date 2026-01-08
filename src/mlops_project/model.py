import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """
    A simple Convolutional Neural Network for MNIST classification.
    """

    def __init__(
            self, 
            in_channels: int = 1, 
            out_channels_1: int = 32, 
            out_channels_2: int = 64, 
            out_channels_3: int = 128,
            kernel_size: int = 3, 
            stride: int = 1,
            dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_1, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=kernel_size, stride=stride)
        self.conv3 = nn.Conv2d(in_channels=out_channels_2, out_channels=out_channels_3, kernel_size=kernel_size, stride=stride)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, 10).
        """
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
    print(f"Model architecture: {model}")
    print(f"#Parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
