import numpy as np


def aggregate_probs(probs, method: str = "mean", topk: int = 5) -> float:
    """Aggregate frame probabilities into a single score in [0, 1]."""

    p = np.array(probs, dtype=float)
    if p.size == 0:
        return 0.0

    if method == "median":
        score = float(np.median(p))
    elif method == "topk":
        k = max(1, min(int(topk), p.size))
        score = float(np.mean(np.sort(p)[-k:]))
    else:
        score = float(np.mean(p))

    # Ensure the score remains within probability bounds.
    return float(np.clip(score, 0.0, 1.0))
