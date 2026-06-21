"""
Minimal FastAPI service for the L8/H512 low-cost intent classifier.

Run from the package root:
    python scripts/serve_fastapi.py --host 127.0.0.1 --port 8000

Then test:
    curl -X POST http://127.0.0.1:8000/predict ^
      -H "Content-Type: application/json" ^
      -d "{\"text\":\"老人发烧了怎么办\"}"
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:
    FastAPI = None
    HTTPException = None
    BaseModel = object
    Field = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

import torch
from transformers import AutoModel, AutoTokenizer, logging as hf_logging


PACKAGE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PACKAGE_DIR / "model"
CONFIG_PATH = MODEL_DIR / "config.json"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
MODEL_PATH = MODEL_DIR / "model.pt"

hf_logging.set_verbosity_error()


class IntentClassifier(torch.nn.Module):
    def __init__(self, base_model_name: str, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))


class ModelState:
    config: Optional[dict] = None
    label_map: Optional[Dict[int, str]] = None
    tokenizer = None
    model: Optional[IntentClassifier] = None
    device: Optional[torch.device] = None


state = ModelState()
FORCE_CPU = False


def load_model(force_cpu: bool = False):
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    label_map = {int(k): v for k, v in json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8")).items()}
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    model = IntentClassifier(config["base_model"], config["hidden_size"], config["num_classes"], config["dropout"])
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    state.config = config
    state.label_map = label_map
    state.tokenizer = tokenizer
    state.model = model
    state.device = device


def ensure_loaded():
    if state.model is None:
        raise RuntimeError("Model is not loaded yet.")


def predict_texts(texts: List[str], top_k: int = 3, batch_size: int = 32):
    ensure_loaded()
    assert state.config is not None
    assert state.label_map is not None
    assert state.tokenizer is not None
    assert state.model is not None
    assert state.device is not None

    results = []
    max_length = state.config["max_seq_length"]
    top_k = max(1, min(top_k, len(state.label_map)))

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoding = state.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(state.device)
            attention_mask = encoding["attention_mask"].to(state.device)
            logits = state.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
            top_values, top_indices = torch.topk(probs, k=top_k, dim=-1)

            for i, text in enumerate(batch_texts):
                intent_id = int(predictions[i].item())
                results.append({
                    "text": text,
                    "intent_id": intent_id,
                    "intent_name": state.label_map[intent_id],
                    "confidence": round(float(confidences[i].item()), 6),
                    "top_k": [
                        {
                            "intent_id": int(top_indices[i][j].item()),
                            "intent_name": state.label_map[int(top_indices[i][j].item())],
                            "probability": round(float(top_values[i][j].item()), 6),
                        }
                        for j in range(top_k)
                    ],
                    "probabilities": {
                        state.label_map[j]: round(float(probs[i][j].item()), 6)
                        for j in range(len(state.label_map))
                    },
                })
    return results


if FastAPI is not None:
    app = FastAPI(title="HC Intent L8/H512 Test Service", version="1.0")

    class PredictRequest(BaseModel):
        text: str = Field(..., min_length=1)
        top_k: int = Field(3, ge=1, le=8)

    class BatchPredictRequest(BaseModel):
        texts: List[str] = Field(..., min_length=1)
        top_k: int = Field(3, ge=1, le=8)
        batch_size: int = Field(32, ge=1, le=256)

    @app.on_event("startup")
    def startup():
        load_model(force_cpu=FORCE_CPU)

    @app.get("/health")
    def health():
        return {
            "ok": state.model is not None,
            "model_id": state.config.get("model_id") if state.config else None,
            "base_model": state.config.get("base_model") if state.config else None,
            "device": str(state.device) if state.device else None,
        }

    @app.post("/predict")
    def predict_one(request: PredictRequest):
        try:
            return predict_texts([request.text], top_k=request.top_k, batch_size=1)[0]
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.post("/predict_batch")
    def predict_batch(request: BatchPredictRequest):
        texts = [text.strip() for text in request.texts if text and text.strip()]
        if not texts:
            raise HTTPException(status_code=400, detail="texts cannot be empty")
        try:
            return {"results": predict_texts(texts, top_k=request.top_k, batch_size=request.batch_size)}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))


def main():
    if _IMPORT_ERROR is not None:
        print("Missing FastAPI runtime dependency.", file=sys.stderr)
        print("Install it first:", file=sys.stderr)
        print(r"  E:\codes\hc-intent\.venv\Scripts\python.exe -m pip install fastapi uvicorn", file=sys.stderr)
        raise SystemExit(1)

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    global FORCE_CPU
    FORCE_CPU = args.device == "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
