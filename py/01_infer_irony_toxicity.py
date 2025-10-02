#!/usr/bin/env python
"""
Infer sarcasm (irony) and toxicity for the TweetEval climate-stance dataset
and export a tidy CSV for R analysis.


Usage:
python py/01_infer_irony_toxicity.py --out data/tweeteval_climate_with_scores.csv
"""
import argparse
import os
from typing import List, Dict


import numpy as np
import pandas as pd
from tqdm import tqdm


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch
from detoxify import Detoxify


HF_DATASET = "cardiffnlp/tweet_eval"
STANCE_CONFIG = "stance_climate"
IRONY_MODEL = "cardiffnlp/twitter-roberta-base-irony"


LABEL_MAP = {0: "none", 1: "against", 2: "favor"}

def chunker(seq: List[str], size: int):
    for pos in range(0, len(seq), size):
        yield seq[pos: pos + size]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def load_irony_pipeline(device: int = -1) -> TextClassificationPipeline:
    tok = AutoTokenizer.from_pretrained(IRONY_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(IRONY_MODEL)
    pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, device=device, return_all_scores=True, truncation=True)
    return pipe

def score_irony(pipe: TextClassificationPipeline, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Return P(irony==1) for each text."""
    probs = []
    for batch in tqdm(list(chunker(texts, batch_size)), desc="Irony", leave=False):
        results = pipe(batch)
        # each result is a list of dicts with 'label' and 'score'
        for r in results:
            # models often label as LABEL_0 (non_irony), LABEL_1 (irony)
            scores = {d["label"]: d["score"] for d in r}
            p_irony = scores.get("LABEL_1", scores.get("1", 0.0))
            probs.append(p_irony)
    return np.array(probs)
        
def score_toxicity(texts: List[str], device: int = -1, batch_size: int = 64) -> Dict[str, np.ndarray]:
    """Return dict of toxicity signal arrays (keys like 'toxicity','insult',...)."""
    mdl = Detoxify("unbiased", device=("cuda" if device >= 0 else "cpu"))
    # Detoxify handles batching internally; call once on all texts
    preds = mdl.predict(texts)
    # ensure dtype is np.array
    return {k: np.asarray(v) for k, v in preds.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tweeteval_climate_with_scores.csv")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print("Loading TweetEval stance_climate …")
    ds_train = load_dataset(HF_DATASET, STANCE_CONFIG, split="train")
    ds_val = load_dataset(HF_DATASET, STANCE_CONFIG, split="validation")
    ds_test = load_dataset(HF_DATASET, STANCE_CONFIG, split="test")

    def to_df(split_ds, name):
        df = pd.DataFrame({"text": split_ds["text"], "label": split_ds["label"]})
        df["split"] = name
        return df

    df = pd.concat([to_df(ds_train, "train"), to_df(ds_val, "validation"), to_df(ds_test, "test")], ignore_index=True)
    df["label_name"] = df["label"].map(LABEL_MAP)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda' if device>=0 else 'cpu'}")

    irony_pipe = load_irony_pipeline(device=device)
    df["sarcasm_prob"] = score_irony(irony_pipe, df["text"].tolist(), batch_size=args.batch)

    tox = score_toxicity(df["text"].tolist(), device=device, batch_size=args.batch)
    for k, v in tox.items():
        df[f"tox_{k}"] = v

    # A single overall toxicity score (use 'toxicity' key if present)
    if "toxicity" in tox:
        df["toxicity"] = tox["toxicity"]
    else:
        # fallback: mean over available toxicity-related dimensions
        tox_cols = [c for c in df.columns if c.startswith("tox_")]
        df["toxicity"] = df[tox_cols].mean(axis=1)

    # Derived convenience fields
    df["is_extreme"] = (df["label_name"].isin(["against", "favor"]).astype(int))
    df["stance_num"] = df["label"].map({0: 0, 1: -1, 2: 1}) # neutral=0, against=-1, favor=+1

    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows to {args.out}")
    

if __name__ == "__main__":
    main()