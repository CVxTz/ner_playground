import torch
from torch.nn.utils.rnn import pad_sequence

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
