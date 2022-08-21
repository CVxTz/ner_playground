from pathlib import Path
from pprint import pprint

import pandas as pd
import torch
from nervaluate import Evaluator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ner_playground.config import CLASSES, INV_LABEL_MAPPING
from ner_playground.data_preparation import prepare_dataset
from ner_playground.models import BaseNerModel
from ner_playground.nlp_utils import decode_labeled_tokens, generate_labeled_tokens

MAX_LEN = 256


if __name__ == "__main__":

    BASE_PATH = Path(__file__).parents[2]

    df = pd.read_csv(BASE_PATH / "data" / "TASTEset.csv")

    full_samples = prepare_dataset(data=df)

    train, val = train_test_split(full_samples, random_state=1337, test_size=0.1)

    model = BaseNerModel(
        lr=5e-5,
    )

    model.eval()

    model_path = BASE_PATH / "models" / "ner-base.ckpt"
    model.load_state_dict(torch.load(model_path)["state_dict"])

    gold_spans = []
    predicted_spans = []

    for sample in tqdm(val):
        text = sample["text"]

        tokens = sample["tokens"][:MAX_LEN]
        prediction_tokens = generate_labeled_tokens(text, labels=[])[:MAX_LEN]

        x = torch.tensor([token.index for token in prediction_tokens], dtype=torch.long)
        x = x.unsqueeze(0)

        with torch.no_grad():
            prediction_score = model(x).squeeze(0)

        _, predicted = torch.max(prediction_score, 1)

        for token, label_index in zip(prediction_tokens, predicted.tolist()):
            token.bio_label = INV_LABEL_MAPPING[label_index]

        gold_spans.append(decode_labeled_tokens(tokens))
        predicted_spans.append(decode_labeled_tokens(prediction_tokens))

    evaluator = Evaluator(gold_spans, predicted_spans, tags=CLASSES)

    # Returns overall metrics and metrics for each tag

    results, results_per_tag = evaluator.evaluate()

    pprint(results["strict"]["f1"])

    # pprint(results_per_tag)
