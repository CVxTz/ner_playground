from pathlib import Path
import json

import pandas as pd
from tqdm import tqdm

from ner_playground.nlp_utils import (
    generate_labeled_tokens,
)


def prepare_dataset(data: pd.DataFrame):

    full_samples = []

    for idx, row in tqdm(data.iterrows()):
        text = row["ingredients"]
        spans = json.loads(row["ingredients_entities"])

        labels = [(span["type"], span["start"], span["end"]) for span in spans]

        tokens = generate_labeled_tokens(text, labels=labels)

        full_samples.append(
            {
                "text": text,
                "tokens": tokens,
            }
        )

    return full_samples


if __name__ == "__main__":

    BASE_PATH = Path(__file__).parents[2] / "data"

    df = pd.read_csv(BASE_PATH / "TASTEset.csv")

    encoded_samples = prepare_dataset(data=df)

    for x in encoded_samples[0]["tokens"]:
        print(x)
