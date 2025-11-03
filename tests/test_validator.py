import math
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.validator import Validator


class DummyDataset(Dataset):
    def __init__(self) -> None:
        self.inputs = torch.stack(
            [torch.zeros(3, 4, 4, dtype=torch.float32), torch.ones(3, 4, 4, dtype=torch.float32)]
        )
        self.labels = torch.tensor([1, 0], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return {"x": self.inputs[index], "y": self.labels[index]}


class MeanBasedModel(torch.nn.Module):
    def forward(self, pixel_values: torch.Tensor):
        mean = pixel_values.mean(dim=(1, 2, 3))
        logits = torch.stack([mean, 1.0 - mean], dim=1)
        return SimpleNamespace(logits=logits)


def test_validator_computes_macro_f1():
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = MeanBasedModel()

    validator = Validator(model=model, device=torch.device("cpu"), amp=False)
    result = validator.evaluate(dataloader)

    assert math.isclose(result.loss, 0.3133, rel_tol=1e-3)
    assert math.isclose(result.macro_f1, 1.0, rel_tol=1e-5)
