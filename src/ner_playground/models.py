from pathlib import Path
import math

import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel

from ner_playground.config import LABEL_MAPPING, PAD_IDX, N_VOCAB

CONFIG_PATH = Path(__file__).parents[2] / "bert_model" / "bert_config.json"
CONFIG = BertConfig.from_json_file(CONFIG_PATH)

import torch
from transformers import get_cosine_schedule_with_warmup


def masked_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, mask):
    y_true = torch.masked_select(y_true, mask)
    y_pred = torch.masked_select(y_pred, mask)

    acc = (y_true == y_pred).double().mean()

    return acc


class BaseModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="valid")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="test")

    def _step(self, batch, batch_idx, name="train"):
        x, y = batch

        y_hat = self(x)

        y_hat = y_hat.reshape(-1, y_hat.size(2))
        y = y.view(-1)

        loss = F.cross_entropy(y_hat, y, reduction="mean")

        _, predicted = torch.max(y_hat, 1)

        mask = x != self.pad_idx
        mask = mask.view(-1)

        acc = masked_accuracy(y, predicted, mask)

        self.log(f"{name}_loss", loss)
        self.log(f"{name}_acc", acc)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_schedulers = {
            "scheduler": get_cosine_schedule_with_warmup(
                optimizer=opt, num_warmup_steps=1000, num_training_steps=7700
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [opt], [lr_schedulers]


class BertNerModel(BaseModel):
    def __init__(
        self,
        n_classes=len(LABEL_MAPPING),
        lr=1e-4,
        pad_idx=PAD_IDX,
        dropout=0.2,
        bert_path=None,
        keep_layers=("embeddings", "encoder", "pooler"),
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.pad_idx = pad_idx

        self.n_classes = n_classes

        CONFIG.hidden_dropout_prob = dropout

        self.bert = BertModel(config=CONFIG)

        for name, param in self.bert.named_parameters():
            if not any(name.startswith(a) for a in keep_layers):
                param.requires_grad = False

        if bert_path:
            state_dict = torch.load(bert_path)
            self.bert.load_state_dict(state_dict)

        self.do = nn.Dropout(p=dropout)

        self.out_linear = nn.Linear(CONFIG.hidden_size, n_classes)

    def forward(self, x):

        mask = (x != self.pad_idx).int()
        x = self.bert(
            x, attention_mask=mask, encoder_attention_mask=mask
        ).last_hidden_state
        # [batch, Seq_len, CONFIG.hidden_size]

        x = self.do(x)

        out = self.out_linear(x)

        return out


class PositionalEncoding(torch.nn.Module):
    #  https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0:, :, 0::2] = torch.sin(position * div_term)
        pe[0:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        x = x + self.pe[:, : x.size(1)] / math.sqrt(self.d_model)

        return self.dropout(x)


class BaseNerModel(BaseModel):
    def __init__(
        self,
        n_classes=len(LABEL_MAPPING),
        lr=1e-4,
        pad_idx=PAD_IDX,
        dropout=0.2,
        d_model=256,
        n_vocab=N_VOCAB
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.dropout = dropout
        self.n_vocab = n_vocab

        self.n_classes = n_classes

        # Text
        self.embeddings = torch.nn.Embedding(self.n_vocab, self.d_model)
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model, dropout=self.dropout
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dropout=self.dropout, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.do = nn.Dropout(p=dropout)

        self.out_linear = nn.Linear(self.d_model, n_classes)

    def forward(self, x):

        mask = (x == self.pad_idx).float()

        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))

        x = self.embeddings(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=mask)

        x = self.do(x)

        out = self.out_linear(x)

        return out


# bert = BertModel(config=CONFIG)
#
# for name, param in bert.named_parameters():
#     print(name)
