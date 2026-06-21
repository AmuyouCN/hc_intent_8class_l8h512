"""
Single-model inference for uer/chinese_roberta_L-8_H-512 + 8-class classifier.
No quantization, no dual-stage, no cache, no rules, no post-processing.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
CONFIG_PATH = MODEL_DIR / "config.json"
MODEL_PATH = MODEL_DIR / "model.pt"


class IntentClassifier(torch.nn.Module):
    def __init__(self, base_model_name, hidden_size, num_classes, dropout):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))


def load_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    label_map = {int(k): v for k, v in json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8")).items()}
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    model = IntentClassifier(config["base_model"], config["hidden_size"], config["num_classes"], config["dropout"])
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return config, label_map, tokenizer, model


def predict(texts, config, label_map, tokenizer, model, device, batch_size=32):
    """Predict intent for a list of texts. Returns list of (intent_id, intent_name, confidence, probs)."""
    from torch.utils.data import DataLoader, Dataset

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length",
                                 max_length=self.max_length, return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0)}

    dataset = TextDataset(texts, tokenizer, config["max_seq_length"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            for i in range(len(preds)):
                pid = preds[i].item()
                conf = probs[i][pid].item()
                all_probs = {label_map[j]: round(probs[i][j].item(), 4) for j in range(len(label_map))}
                results.append((pid, label_map[pid], round(conf, 4), all_probs))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*", help="Text(s) to classify")
    parser.add_argument("--file", help="File with one text per line")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    texts = list(args.text)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts.extend(line.strip() for line in f if line.strip())
    if not texts:
        print("No input texts. Pass text as argument or use --file.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, label_map, tokenizer, model = load_model(device)
    results = predict(texts, config, label_map, tokenizer, model, device, args.batch_size)

    for text, (intent_id, intent_name, conf, probs) in zip(texts, results):
        print(f"Text: {text}")
        print(f"  Intent: {intent_id} ({intent_name}), Confidence: {conf}")
        print(f"  Probabilities: {probs}")
        print()


if __name__ == "__main__":
    main()
