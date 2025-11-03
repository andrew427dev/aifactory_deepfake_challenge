# src/evaluation/threshold.py
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="probs csv path: [filename, prob]")
    ap.add_argument("--labels", required=True, help="labels csv path: [filename, label]")
    args = ap.parse_args()

    probs = pd.read_csv(args.oof)
    labels = pd.read_csv(args.labels)
    df = probs.merge(labels, on="filename", how="inner")
    y_true = df["label"].astype(int).values
    p = df["prob"].astype(float).values

    best_thr, best_f1 = 0.5, -1
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (p >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro")
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    print(f"[Best] threshold={best_thr:.4f}, macro_f1={best_f1:.5f}")

if __name__ == "__main__":
    main()
