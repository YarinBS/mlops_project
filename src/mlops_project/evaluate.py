import torch
from torch.utils.data import DataLoader
import typer

from mlops_project.model import MyAwesomeModel
from mlops_project.data import corrupt_mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """
    Evaluate the trained model on the test dataset.

    Parameters:
    model_checkpoint (str): Path to the trained model checkpoint.
    """

    state_dict = torch.load(model_checkpoint, map_location=DEVICE)
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(state_dict)

    _, test_set = corrupt_mnist()
    test_loader = DataLoader(test_set, batch_size=64)

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


def main():
    typer.run(evaluate)


if __name__ == "__main__":
    main()
