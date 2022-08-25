from pathlib import Path

from transformers import BertModel, BertTokenizer

BASE_PATH = Path(__file__).parents[1] / "bert_model"

BASE_PATH.mkdir(exist_ok=True)

model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

for name, param in model.named_parameters():
    print(name)

model.config.to_json_file(BASE_PATH / "bert_config.json")
model.save_pretrained(BASE_PATH)
tokenizer.save_pretrained(BASE_PATH)
