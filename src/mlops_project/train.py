import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import typer

from mlops_project.model import MyAwesomeModel
from mlops_project.data import corrupt_mnist


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(lr: float = 1e-3, epochs: int = 20) -> None:
    """
    Train the MyAwesomeModel on the corrupted MNIST dataset.

    Parameters:
    lr (float): Learning rate for the optimizer.
    epochs (int): Number of training epochs.
    """

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_loader = DataLoader(train_set, batch_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses, train_accuracies = [], []

    for epoch in range(epochs):
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

        print(f"Epoch {epoch + 1}/{epochs}, Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")

    print("Training complete")
    torch.save(model.state_dict(), "models/trained_model.pth")

    # Plotting training loss
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.savefig("reports/figures/training_loss.png")


def main():
    typer.run(train)

if __name__ == "__main__":
    main()
