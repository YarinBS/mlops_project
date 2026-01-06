import matplotlib.pyplot as plt
import torch
import typer

from src.mlops_project.data import corrupt_mnist
from src.mlops_project.model import MyAwesomeModel

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.command()
def train(
    lr: float = 1e-3,
    epochs: int = 20,
) -> None:
    """Train a model on MNIST."""

    model = MyAwesomeModel().to(DEVICE)
    train_loader, _ = corrupt_mnist()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train loss: {train_loss:.4f}")

    print("Training complete")
    torch.save(model.state_dict(), "trained_model.pth")

    # Plotting training loss
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.savefig("training_loss.png")


@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""

    state_dict = torch.load(model_checkpoint, map_location=DEVICE)
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(state_dict)

    _, test_loader = corrupt_mnist()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    app()
