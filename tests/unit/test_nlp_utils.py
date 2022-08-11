from faker import Faker

from ner_playground.nlp_utils import (
    CLS,
    SEP,
    TOKENIZER,
    generate_labeled_tokens,
    tokenize,
    decode_labeled_tokens
)

fake = Faker()


def test_tokenize():
    for _ in range(5):
        text = fake.text()
        tokens = tokenize(text)
        assert (
            TOKENIZER.decode([token.index for token in tokens[1:-1]]).replace("\n", " ")
            == " ".join(text.split()).lower()
        )


def test_tokenize_2():
    text = "Hello, Its me, Mario.\n\nLuigi"
    tokens = tokenize(text)
    tokens_txt = [token.token for token in tokens]
    assert tokens_txt == [
        "[CLS]",
        "hello",
        ",",
        "its",
        "me",
        ",",
        "mario",
        ".",
        "luigi",
        "[SEP]",
    ]


def test_generate_labeled_tokens():
    text_1 = "Hello, my name is mario."
    labels_1 = [("A", 0, 6), ("A", 7, 9), ("X", 15, 23)]
    expected_label_sequence = ["B-A", "I-A", "B-A", "O", "B-X", "I-X", "O"]
    tokens = generate_labeled_tokens(text=text_1, labels=labels_1)[1:-1]
    assert [token.bio_label for token in tokens] == expected_label_sequence

    text_2 = "Hello, my name is mario."
    labels_2 = [("A", 5, 6), ("A", 7, 9), ("X", 15, 24)]
    expected_label_sequence = ["O", "B-A", "B-A", "O", "B-X", "I-X", "I-X"]
    tokens = generate_labeled_tokens(text=text_2, labels=labels_2)[1:-1]
    assert [token.bio_label for token in tokens] == expected_label_sequence

    text_3 = "Hello, my name is mario."
    labels_3 = [("A", 7, 9), ("X", 15, 24)]
    expected_label_sequence = ["O", "O", "B-A", "O", "B-X", "I-X", "I-X"]
    tokens = generate_labeled_tokens(text=text_3, labels=labels_3)[1:-1]
    assert [token.bio_label for token in tokens] == expected_label_sequence

    text_4 = "Hello, my name is mario."
    labels_4 = [("A", 7, 9), ("X", 15, 24)]
    tokens = generate_labeled_tokens(text=text_4, labels=labels_4)
    assert tokens[0].token == CLS
    assert tokens[-1].token == SEP


def test_decode_labeled_tokens():
    text_1 = "Hello, my name is mario."
    labels_1 = [("A", 0, 6), ("A", 7, 9), ("X", 15, 23)]
    tokens_1 = generate_labeled_tokens(text=text_1, labels=labels_1)
    labels_decoded_1, _ = decode_labeled_tokens(tokens_1, text_1)
    assert labels_1 == labels_decoded_1

    text_2 = "Hello, my name is mario."
    labels_2 = [("A", 5, 6), ("A", 7, 9), ("X", 15, 24)]
    tokens_2 = generate_labeled_tokens(text=text_2, labels=labels_2)
    labels_decoded_2, _ = decode_labeled_tokens(tokens_2, text_2)
    assert labels_2 == labels_decoded_2

    text_3 = "Hello, my name is mario."
    labels_3 = [("A", 7, 9), ("X", 15, 24)]
    tokens_3 = generate_labeled_tokens(text=text_3, labels=labels_3)
    labels_decoded_3, _ = decode_labeled_tokens(tokens_3, text_3)
    assert labels_3 == labels_decoded_3

    text_4 = "Hello, my name is mario."
    labels_4 = [("A", 7, 9), ("X", 15, 24)]
    tokens_4 = generate_labeled_tokens(text=text_4, labels=labels_4)
    labels_decoded_4, _ = decode_labeled_tokens(tokens_4, text_4)
    assert labels_4 == labels_decoded_4
