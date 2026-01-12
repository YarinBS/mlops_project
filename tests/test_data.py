import torch
from torch.utils.data import TensorDataset
from unittest.mock import patch

from mlops_project.data import corrupt_mnist


def test_my_dataset():

    mock_train = TensorDataset(
        torch.randn(50000, 1, 28, 28),
        torch.randint(0, 10, (50000,))
    )
    mock_test = TensorDataset(
        torch.randn(5000, 1, 28, 28),
        torch.randint(0, 10, (5000,))
    )

    with patch('mlops_project.data.corrupt_mnist', return_value=(mock_train, mock_test)):
        train, test = corrupt_mnist()
        assert len(train) == 50000
        assert len(test) == 5000

        for dataset in [train, test]:
            for x, y in dataset:
                assert x.shape == (1, 28, 28)
                assert y in range(10)
        
        train_targets = torch.unique(train.tensors[1])
        assert (train_targets == torch.arange(0,10)).all()
        test_targets = torch.unique(test.tensors[1])
        assert (test_targets == torch.arange(0,10)).all()
