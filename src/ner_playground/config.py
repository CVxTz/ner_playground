from pathlib import Path

from transformers import BertTokenizerFast


print(Path(__file__).parents[2])

TOKENIZER_PATH = Path(__file__).parents[2] / "bert_model"

TOKENIZER = BertTokenizerFast.from_pretrained(str(TOKENIZER_PATH))
CLASSES = [
    "FOOD",
    "QUANTITY",
    "UNIT",
    "PROCESS",
    "PHYSICAL_QUALITY",
    "COLOR",
    "TASTE",
    "PURPOSE",
    "PART",
]

LABEL_MAPPING = {
    "O": 0,
}
i = 1
for c in CLASSES:
    LABEL_MAPPING[f"B-{c}"] = i
    LABEL_MAPPING[f"I-{c}"] = i + 1


INV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

CLS = "[CLS]"
PAD = "[PAD]"
SEP = "[SEP]"

PAD_IDX = TOKENIZER.pad_token_id
CLS_IDX = TOKENIZER.cls_token_id
SEP_IDX = TOKENIZER.sep_token_id

print(TOKENIZER.encode_plus("Hello word!", return_offsets_mapping=True))
