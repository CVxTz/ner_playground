import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from functools import partial
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ner_playground.config import PAD_IDX
from ner_playground.data_preparation import prepare_dataset
from ner_playground.models import NerModel

MAX_LEN = 256


class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]["tokens"]

        window_tokens = tokens[:MAX_LEN]

        x = torch.tensor([token.index for token in window_tokens], dtype=torch.long)
        y = torch.tensor([token.bio_idx for token in window_tokens], dtype=torch.long)

        return x, y


def generate_batch(data_batch, pad_idx):
    src, trg = [], []
    for (src_item, trg_item) in data_batch:
        src.append(src_item)
        trg.append(trg_item)
    src = pad_sequence(src, padding_value=pad_idx, batch_first=True)
    trg = pad_sequence(trg, padding_value=pad_idx, batch_first=True)

    return src, trg


if __name__ == "__main__":

    batch_size = 32
    epochs = 128

    BASE_PATH = Path(__file__).parents[2]
    BERT_PATH = BASE_PATH / "bert_model" / "pytorch_model.bin"

    df = pd.read_csv(BASE_PATH / "data" / "TASTEset.csv")

    full_samples = prepare_dataset(data=df)

    train, val = train_test_split(full_samples, random_state=1337, test_size=0.1)

    train_data = Dataset(samples=train)
    val_data = Dataset(samples=val)

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True,
        collate_fn=partial(generate_batch, pad_idx=PAD_IDX),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True,
        collate_fn=partial(generate_batch, pad_idx=PAD_IDX),
    )

    model = NerModel(
        lr=5e-5,
        bert_path=BERT_PATH,
        # keep_layers=("encoder.layer.11", ),
    )

    # model_path = Path(__file__).parents[1] / "models" / "segmenter_no_weights.ckpt"
    # model.load_state_dict(torch.load(model_path)["state_dict"])

    logger = TensorBoardLogger(save_dir=str(BASE_PATH), name="logs", version="ner")

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_acc",
        mode="max",
        dirpath=BASE_PATH / "models",
        filename="ner",
        save_weights_only=True,
        every_n_epochs=16,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],  # checkpoint_callback,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_loader, val_loader)
