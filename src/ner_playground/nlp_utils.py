import re
from typing import Dict, List, Tuple

from ner_playground.config import LABEL_MAPPING, TOKENIZER, SEP, SEP_IDX, CLS, CLS_IDX


class Word:
    def __init__(self, word: str, index: int, start_index: int, end_index: int):
        self.word = word
        self.index = index
        self.start_index = start_index
        self.end_index = end_index

    def __repr__(self):
        return f"W: {self.word} / I: {self.index} / S: {self.start_index} / E: {self.end_index}"


class Token:
    def __init__(
        self,
        token: str,
        index: int,
        start_index: int,
        end_index: int,
        raw_label: str = "O",
        bio_label: str = "O",
        weight: float = 1,
    ):
        self.token = token
        self.index = index
        self.start_index = start_index
        self.end_index = end_index
        self.raw_label = raw_label
        self.bio_label = bio_label
        self.weight = weight

    def __repr__(self):
        return (
            f"T: {self.token} / "
            f"I: {self.index} / "
            f"S: {self.start_index} / "
            f"E: {self.end_index} / "
            f"RL: {self.raw_label} / "
            f"BIO: {self.bio_label} / "
            f"W: {self.weight}"
        )

    @property
    def bio_idx(self):
        return LABEL_MAPPING[self.bio_label]

    def as_dict(self):
        return {
            "token": self.token,
            "index": self.index,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "raw_label": self.raw_label,
            "bio_label": self.bio_label,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, as_dict: Dict):
        return cls(**as_dict)


def tokenize(text: str):
    encoded = TOKENIZER.encode_plus(text, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    tokens = TOKENIZER.convert_ids_to_tokens(ids)

    tokens = [
        Token(token=token, index=index, start_index=offset[0], end_index=offset[1])
        for token, index, offset in zip(tokens, ids, offsets)
    ]

    return tokens


def most_frequent(list_of_labels):
    return max(set(list_of_labels), key=list_of_labels.count)


def generate_labeled_tokens(text: str, labels: List[Tuple[str, int, int]]):
    tokens = tokenize(text=text)

    char_label = ["O"] * len(text)

    for label, discourse_start, discourse_end in labels:
        char_label[discourse_start:discourse_end] = [label] * (
            discourse_end - discourse_start
        )

    for i, token in enumerate(tokens):
        if token.start_index != token.end_index:
            token.raw_label = most_frequent(
                char_label[token.start_index : token.end_index]
            )

    # BIO labels
    for i, token in enumerate(tokens):
        if token.raw_label != "O":
            if i == 0:
                token.bio_label = "B-" + token.raw_label

            else:
                if tokens[i - 1].raw_label == tokens[i].raw_label:
                    token.bio_label = "I-" + token.raw_label
                else:
                    token.bio_label = "B-" + token.raw_label
        else:
            token.bio_label = token.raw_label

    return tokens


def group_tokens_by_entity(tokens: List[Token]):
    """
    List to List[List[Token]]

    :param tokens:
    :return:
    """
    block_tokens = []
    for i, token in enumerate(tokens):
        if token.bio_label == "O" or token.start_index == token.end_index == 0:
            continue
        elif i == 0:
            block_tokens.append([token])
        elif (
            tokens[i].bio_label.split("-")[0] == "B"
            or tokens[i - 1].bio_label.split("-")[-1]
            != tokens[i].bio_label.split("-")[-1]
        ):
            block_tokens.append([token])
        else:
            block_tokens[-1].append(token)

    return block_tokens
