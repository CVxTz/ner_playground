import torch

from ner_playground.models import NerModel


def test_model():
    model = NerModel(n_classes=20)

    x = torch.arange(0, 20, dtype=torch.long).view(1, 20)

    y = model(x)
    assert y.size() == (1, 20, model.n_classes)
