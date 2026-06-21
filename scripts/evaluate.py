"""
Evaluate model on the current independent standard evaluation dataset.
Outputs overall metrics and per-class metrics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
CONFIG_PATH = MODEL_DIR / "config.json"
MODEL_PATH = MODEL_DIR / "model.pt"

EVAL_DATASETS = {
    "standard": DATA_DIR / "eval_standard.csv",
}
EMERGENCY_CLASS_ID = 5


class IntentClassifier(torch.nn.Module):
    def __init__(self, base_model_name, hidden_size, num_classes, dropout):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(outputs.last_hidden_state[:, 0, :]))


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts, self.labels, self.tokenizer, self.max_length = texts, labels, tokenizer, max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True, padding="max_length",
                             max_length=self.max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(int(self.labels[idx]), dtype=torch.long)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    label_map = {int(k): v for k, v in json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8")).items()}
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    model = IntentClassifier(config["base_model"], config["hidden_size"], config["num_classes"], config["dropout"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir) if args.output_dir else MODEL_DIR.parent / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for ds_name, ds_path in EVAL_DATASETS.items():
        df = pd.read_csv(ds_path)
        dataset = IntentDataset(df["text"].astype(str).tolist(), df["intent_id"].astype(int).tolist(),
                                tokenizer, config["max_seq_length"])
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        labels, preds, confidences = [], [], []
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                probs = torch.softmax(logits, dim=-1)
                preds.extend(torch.argmax(probs, dim=-1).cpu().tolist())
                confidences.extend(torch.max(probs, dim=-1).values.cpu().tolist())
                labels.extend(batch["label"].tolist())

        labels_arr, preds_arr = np.array(labels), np.array(preds)
        recalls = [(preds_arr[labels_arr == i] == i).mean() if (labels_arr == i).any() else 0.0
                   for i in range(config["num_classes"])]
        non_emerg = labels_arr != EMERGENCY_CLASS_ID
        fp_rate = (preds_arr[non_emerg] == EMERGENCY_CLASS_ID).sum() / non_emerg.sum() if non_emerg.sum() else 0.0

        metrics = {
            "dataset": ds_name,
            "samples": len(labels), "accuracy": round(accuracy_score(labels_arr, preds_arr), 4),
            "macro_f1": round(f1_score(labels_arr, preds_arr, average="macro", zero_division=0), 4),
            "macro_precision": round(precision_score(labels_arr, preds_arr, average="macro", zero_division=0), 4),
            "macro_recall": round(recall_score(labels_arr, preds_arr, average="macro", zero_division=0), 4),
            "min_recall": round(float(min(recalls)), 4),
            "bottom3_avg": round(float(np.mean(sorted(recalls)[:3])), 4),
            "emergency_recall": round(float(recalls[EMERGENCY_CLASS_ID]), 4),
            "non_emerg_to_emerg_fp_rate": round(float(fp_rate), 4),
            "avg_confidence": round(float(np.mean(confidences)), 4),
            "device": str(device),
        }

        ds_dir = output_dir / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        per_class = []
        for i in range(config["num_classes"]):
            tp = int(((preds_arr == i) & (labels_arr == i)).sum())
            pred_count = int((preds_arr == i).sum())
            support = int((labels_arr == i).sum())
            precision = tp / pred_count if pred_count else 0.0
            recall = recalls[i]
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class.append({
                "intent_id": i,
                "intent_name": label_map[i],
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "support": support,
                "predicted": pred_count,
            })
        pd.DataFrame(per_class).to_csv(ds_dir / "per_class.csv", index=False, encoding="utf-8-sig")

        print(f"{ds_name}: acc={metrics['accuracy']:.4f} minR={metrics['min_recall']:.2f} "
              f"bot3={metrics['bottom3_avg']:.2f} emergR={metrics['emergency_recall']:.2f}")
        summary.append({"dataset": ds_name, **metrics})

    pd.DataFrame(summary).to_csv(output_dir / "summary.csv", index=False)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
