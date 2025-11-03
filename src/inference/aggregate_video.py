import numpy as np
def aggregate_probs(probs, method="mean", topk=5, threshold=0.5):
    p = np.array(probs, dtype=float)
    if p.size == 0: return 0
    if method == "median": score = np.median(p)
    elif method == "topk": score = np.mean(np.sort(p)[-max(1, min(topk, len(p))):])
    else: score = float(np.mean(p))
    return 1 if score >= float(threshold) else 0
