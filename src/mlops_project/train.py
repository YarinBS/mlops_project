import hydra
import logging
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import os
import torch
from torch.utils.data import DataLoader

from mlops_project.model import MyAwesomeModel
from mlops_project.data import corrupt_mnist

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="hydra_conf", config_name="config")
def train(config: DictConfig) -> None:
    """
    Train the MyAwesomeModel on the corrupted MNIST dataset.

    Parameters:
    lr (float): Learning rate for the optimizer.
    epochs (int): Number of training epochs.
    """

    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    torch.manual_seed(config.seed)

    model = MyAwesomeModel(
        in_channels=config.in_channels,
        out_channels_1=config.out_channels_1,
        out_channels_2=config.out_channels_2,
        out_channels_3=config.out_channels_3,
        kernel_size=config.kernel_size,
        stride=config.stride,
        dropout=config.dropout,
    ).to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_loader = DataLoader(train_set, batch_size=config.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses, train_accuracies = [], []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)  # accumulating loss, taking into account the batch size
            _, preds = torch.max(output, 1)
            train_acc += torch.sum(preds == y).item()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        train_acc /= len(train_loader.dataset)
        train_accuracies.append(train_acc)

        logger.info(f"Epoch {epoch + 1}/{config.epochs}, Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")

    logger.info("Training complete")
    torch.save(model.state_dict(), config.model_save_path)

    # Plotting training loss
    plt.plot(range(1, config.epochs + 1), train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.savefig(os.path.join(config.figures_dir, "training_loss.png"))


if __name__ == "__main__":
    train()
