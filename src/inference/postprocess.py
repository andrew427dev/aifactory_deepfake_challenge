"""Post-processing utilities for inference outputs."""


def binarize(score: float, threshold: float) -> int:
    """Convert a probability score into a binary label."""

    return int(float(score) >= float(threshold))
