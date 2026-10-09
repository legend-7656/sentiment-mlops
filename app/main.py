import os
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

MODEL_DIR = os.getenv("MODEL_DIR", "model")
LABELS = ["negative", "neutral", "positive"]
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
    tok.enable_truncation(max_length=128)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
    state["tok"] = tok
    state["sess"] = ort.InferenceSession(
        f"{MODEL_DIR}/model.onnx", opts, providers=["CPUExecutionProvider"])
    yield
    state.clear()


app = FastAPI(title="Sentiment API", version="1.0", lifespan=lifespan)


class One(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class Many(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


def predict(texts: list[str]) -> list[dict]:
    enc = state["tok"].encode_batch(texts)
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    logits = state["sess"].run(None, {"input_ids": ids, "attention_mask": mask})[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return [{"label": LABELS[int(p.argmax())],
             "scores": {l: round(float(s), 4) for l, s in zip(LABELS, p)}}
            for p in probs]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_one(body: One):
    return predict([body.text])[0]


@app.post("/predict/batch")
def predict_batch(body: Many):
    return {"results": predict(body.texts)}