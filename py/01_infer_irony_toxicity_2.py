#!/usr/bin/env python
"""
Infer sarcasm (irony) and toxicity for the TweetEval climate-stance dataset
and export a tidy CSV for R analysis.

Usage:
  python py/01_infer_irony_toxicity_2.py --out data/tweeteval_climate_with_scores.csv
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

def load_irony_pipeline(device: int = -1):
    """
    Returns: (pipeline, irony_label)
    where irony_label is the exact string used by the model for the irony class.
    """
    tok = AutoTokenizer.from_pretrained(IRONY_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(IRONY_MODEL)

    # Robustly find which label means "irony"
    # (works whether labels are 'LABEL_0/1' or 'non_irony/irony', etc.)
    id2label = {int(k): v for k, v in mdl.config.id2label.items()}
    irony_label = None
    for _, name in id2label.items():
        low = name.lower()
        if "irony" in low and "non" not in low:  # prefer the positive irony class
            irony_label = name
            break
    if irony_label is None:
        # Fallback to index 1 if not found (most binary models put the positive at 1)
        irony_label = id2label.get(1, list(id2label.values())[-1])

    pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, device=device, truncation=True)
    return pipe, irony_label

def score_irony(pipe: TextClassificationPipeline, texts: List[str], irony_label: str, batch_size: int = 64) -> np.ndarray:
    """Return P(irony==1) for each text, robust to label naming."""
    probs = []
    for batch in tqdm(list(chunker(texts, batch_size)), desc="Irony", leave=False):
        # Use top_k=None (replacement for return_all_scores=True) and set max_length to avoid warnings
        results = pipe(batch, top_k=None, truncation=True, max_length=128)
        # each result is a list of dicts with 'label' and 'score'
        for r in results:
            scores = {d["label"]: d["score"] for d in r}
            if irony_label in scores:
                p_irony = float(scores[irony_label])
            else:
                # Last-resort fallback: if exactly two classes and 'non' label is present, take 1 - P(non)
                # or else take the highest-prob label that contains 'irony'
                non_keys = [k for k in scores.keys() if "non" in k.lower()]
                if len(scores) == 2 and non_keys:
                    p_irony = 1.0 - float(scores[non_keys[0]])
                else:
                    cand = [k for k in scores.keys() if "irony" in k.lower()]
                    p_irony = float(scores[cand[0]]) if cand else float(max(scores.values()))
            probs.append(p_irony)

    probs_array = np.array(probs)

    # Debug output
    print(f"\n=== Irony Scoring Debug ===")
    print(f"Total texts processed: {len(probs_array)}")
    print(f"Min sarcasm prob: {probs_array.min():.4f}")
    print(f"Max sarcasm prob: {probs_array.max():.4f}")
    print(f"Mean sarcasm prob: {probs_array.mean():.4f}")
    print(f"Std sarcasm prob: {probs_array.std():.4f}")
    print(f"Number of zeros: {(probs_array == 0).sum()}")
    print(f"Number > 0.1: {(probs_array > 0.1).sum()}")
    print(f"Number > 0.5: {(probs_array > 0.5).sum()}")

    print(f"\nSample predictions (first 5):")
    for i in range(min(5, len(texts))):
        print(f"  Text: {texts[i][:80]}...")
        print(f"  Sarcasm prob: {probs_array[i]:.4f}")

    return probs_array

def score_toxicity(texts: List[str], device: int = -1, batch_size: int = 64) -> Dict[str, np.ndarray]:
    """Return dict of toxicity signal arrays (keys like 'toxicity','insult',...)."""
    # Detoxify accepts 'cpu'/'cuda' strings; map device int to string
    dev = "cuda" if device >= 0 and torch.cuda.is_available() else "cpu"
    mdl = Detoxify("unbiased", device=dev)
    preds = mdl.predict(texts)
    return {k: np.asarray(v) for k, v in preds.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tweeteval_climate_with_scores.csv")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading TweetEval stance_climate …")
    ds_train = load_dataset(HF_DATASET, STANCE_CONFIG, split="train")
    ds_val   = load_dataset(HF_DATASET, STANCE_CONFIG, split="validation")
    ds_test  = load_dataset(HF_DATASET, STANCE_CONFIG, split="test")

    def to_df(split_ds, name):
        df = pd.DataFrame({"text": split_ds["text"], "label": split_ds["label"]})
        df["split"] = name
        return df

    df = pd.concat(
        [to_df(ds_train, "train"), to_df(ds_val, "validation"), to_df(ds_test, "test")],
        ignore_index=True
    )
    df["label_name"] = df["label"].map(LABEL_MAP)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda' if device>=0 else 'cpu'}")

    irony_pipe, irony_label = load_irony_pipeline(device=device)
    print(f"Resolved irony label: {irony_label}")
    df["sarcasm_prob"] = score_irony(irony_pipe, df["text"].tolist(), irony_label, batch_size=args.batch)

    tox = score_toxicity(df["text"].tolist(), device=device, batch_size=args.batch)
    for k, v in tox.items():
        df[f"tox_{k}"] = v

    # Overall toxicity score
    if "toxicity" in tox:
        df["toxicity"] = tox["toxicity"]
    else:
        tox_cols = [c for c in df.columns if c.startswith("tox_")]
        df["toxicity"] = df[tox_cols].mean(axis=1)

    # Derived fields
    df["is_extreme"] = (df["label_name"].isin(["against", "favor"]).astype(int))
    df["stance_num"] = df["label"].map({0: 0, 1: -1, 2: 1})  # neutral=0, against=-1, favor=+1

    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df):,} rows to {args.out}")

if __name__ == "__main__":
    main()
