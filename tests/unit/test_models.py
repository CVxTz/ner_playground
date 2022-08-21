import torch

from ner_playground.models import BertNerModel


def test_model():
    model = BertNerModel(n_classes=20)

    x = torch.arange(0, 20, dtype=torch.long).view(1, 20)

    y = model(x)
    assert y.size() == (1, 20, model.n_classes)
