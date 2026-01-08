import pytest
import torch

from mlops_project.model import MyAwesomeModel


def test_model_output():
    model = MyAwesomeModel()
    sample_input = torch.randn(1, 1, 28, 28)
    output = model(sample_input)
    assert output.shape == (1, 10), "Output shape is incorrect"

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(RuntimeError):
        model(torch.randn(1,2,3))
    with pytest.raises(RuntimeError, match='mat1 and mat2'):
        model(torch.randn(1,1,28,30))

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10), f"Output shape is incorrect for batch size {batch_size}"