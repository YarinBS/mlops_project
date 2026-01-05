import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset


def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    
    train_x = torch.cat(
        [torch.load(f"data/corruptmnist/train_images_{i}.pt") for i in range(6)]
    )
    train_y = torch.cat(
        [torch.load(f"data/corruptmnist/train_target_{i}.pt") for i in range(6)]
    )

    test_x = torch.load("data/corruptmnist/test_images.pt")
    test_y = torch.load("data/corruptmnist/test_target.pt")
    
    train_loader = DataLoader(
        TensorDataset(train_x.unsqueeze(1).float(), train_y.long()), 
        batch_size=64, 
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(test_x.unsqueeze(1).float(), test_y.long()), 
        batch_size=64, 
        shuffle=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train, test = corrupt_mnist()
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    # Display some samples from the training set
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(train[i].squeeze(), cmap="gray")
        axes[i].axis("off")
    plt.savefig("corrupt_mnist_samples.png")

    # Display some samples from the test set
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(test[i].squeeze(), cmap="gray")
        axes[i].axis("off")
    plt.savefig("corrupt_mnist_test_samples.png")
