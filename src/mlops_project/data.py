import os

import torch
from torch.utils.data import TensorDataset
import typer


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor to have zero mean and unit variance.

    Parameters:
    tensor (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Normalized tensor.
    """
    return (tensor - tensor.mean()) / tensor.std()


def preprocess_data(raw_data_dir: str, processed_data_dir: str) -> None:
    """
    Preprocess raw data and save the processed data.

    Parameters:
    raw_data_dir (str): Path to the raw data directory.
    processed_data_dir (str): Path to save the processed data.
    """

    train_x = torch.cat([torch.load(os.path.join(raw_data_dir, f"train_images_{i}.pt")) for i in range(10)])
    train_x = _normalize(train_x.unsqueeze(1).float())

    train_y = torch.cat([torch.load(os.path.join(raw_data_dir, f"train_target_{i}.pt")) for i in range(10)])
    train_y = train_y.long()

    test_x = torch.load(os.path.join(raw_data_dir, "test_images.pt"))
    test_x = _normalize(test_x.unsqueeze(1).float())

    test_y = torch.load(os.path.join(raw_data_dir, "test_target.pt"))
    test_y = test_y.long()

    torch.save(train_x, os.path.join(processed_data_dir, "train_images.pt"))
    torch.save(train_y, os.path.join(processed_data_dir, "train_target.pt"))
    torch.save(test_x, os.path.join(processed_data_dir, "test_images.pt"))
    torch.save(test_y, os.path.join(processed_data_dir, "test_target.pt"))


def corrupt_mnist() -> tuple[TensorDataset, TensorDataset]:
    """
    Load the processed corrupted MNIST dataset.

    Returns:
    tuple[DataLoader, DataLoader]: Training and testing data loaders.
    """

    train_x = torch.load("data/processed/train_images.pt")
    train_y = torch.load("data/processed/train_target.pt")
    test_x = torch.load("data/processed/test_images.pt")
    test_y = torch.load("data/processed/test_target.pt")

    train_set = TensorDataset(train_x, train_y)
    test_set = TensorDataset(test_x, test_y)

    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
