from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.eval()

for name, param in model.named_parameters():
    print(name)

model.config.to_json_file("bert_model/bert_config.json")
model.save_pretrained("bert_model")

tokenizer.save_pretrained("bert_model")
