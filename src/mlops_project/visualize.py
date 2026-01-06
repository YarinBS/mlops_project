import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
import typer

from mlops_project.model import MyAwesomeModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """
    Visualize embeddings of the test dataset using t-SNE.

    Parameters:
    model_checkpoint (str): Path to the trained model checkpoint.
    figure_name (str): Name of the output figure file.
    """
    model: torch.nn.Module = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    # Replacing the final fully-connected layer with identity to get embeddings
    model.fc = torch.nn.Identity()

    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)
    test_loader = DataLoader(test_dataset, batch_size=32)

    embeddings, targets = [], []
    with torch.inference_mode():
        for images, target in test_loader:
            images = images.to(DEVICE)
            predictions = model(images)
            embeddings.append(predictions.cpu())
            targets.append(target)

        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    # Apply PCA if embeddings are high-dimensional (reduces t-SNE computation time), and then t-SNE
    if embeddings.shape[1] > 500:
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    # Finally, create a scatter plot of the embeddings
    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


def main():
    typer.run(visualize)


if __name__ == "__main__":
    main()
