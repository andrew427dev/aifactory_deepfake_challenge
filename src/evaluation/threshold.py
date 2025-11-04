"""Threshold search utility for macro F1 maximization."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from decimal import Decimal, getcontext
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search best threshold based on macro F1")
    parser.add_argument("--oof", required=True, help="probabilities CSV path: [filename, prob]")
    parser.add_argument("--labels", required=True, help="labels CSV path: [filename, label]")
    parser.add_argument(
        "--grid",
        nargs=3,
        type=float,
        metavar=("START", "END", "STEP"),
        help="threshold grid definition (defaults to 0.01 0.99 0.01)",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="output JSON path (default: reports/threshold_<date>.json)",
    )
    parser.add_argument(
        "--write-yaml",
        dest="write_yaml",
        type=str,
        help="optional infer.yaml path to update postprocess.threshold",
    )
    return parser.parse_args()


def build_threshold_grid(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("Grid step must be positive")
    if end < start:
        raise ValueError("Grid end must be greater than or equal to start")
    getcontext().prec = 12
    start_dec = Decimal(str(start))
    end_dec = Decimal(str(end))
    step_dec = Decimal(str(step))
    thresholds: List[float] = []
    current = start_dec
    epsilon = Decimal("1e-9")
    while current <= end_dec + epsilon:
        thresholds.append(float(current))
        current += step_dec
    return thresholds


def evaluate_thresholds(y_true: Sequence[int], probs: Sequence[float], thresholds: Sequence[float]) -> Tuple[List[Tuple[float, float]], float, float]:
    results: List[Tuple[float, float]] = []
    best_threshold = 0.0
    best_score = -1.0
    y_true_arr = np.asarray(y_true, dtype=int)
    prob_arr = np.asarray(probs, dtype=float)
    for thr in thresholds:
        preds = (prob_arr >= thr).astype(int)
        score = float(f1_score(y_true_arr, preds, average="macro"))
        results.append((thr, score))
        if score > best_score:
            best_score = score
            best_threshold = float(thr)
    return results, best_threshold, best_score


def format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def sample_grid_points(
    results: Sequence[Tuple[float, float]],
    start: float,
    end: float,
    best_threshold: float,
    best_score: float,
) -> List[dict]:
    if not results:
        return []
    thresholds = np.array([thr for thr, _ in results], dtype=float)
    scores = np.array([score for _, score in results], dtype=float)
    sample_targets = np.arange(0.0, 1.0001, 0.05)
    sampled: dict[float, float] = {}
    for target in sample_targets:
        if start - 1e-9 <= target <= end + 1e-9:
            idx = int(np.abs(thresholds - target).argmin())
            key = round(float(thresholds[idx]), 6)
            sampled[key] = float(scores[idx])
    best_key = round(float(best_threshold), 6)
    if best_key not in sampled:
        sampled[best_key] = float(best_score)
    return [
        {"threshold": thr, "macro_f1": sampled[thr]}
        for thr in sorted(sampled.keys())
    ]


def update_threshold_yaml(yaml_path: Path, new_value: float) -> None:
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    postprocess_indent = None
    start_index = None
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("postprocess:"):
            postprocess_indent = len(line) - len(stripped)
            start_index = idx + 1
            break
    if postprocess_indent is None or start_index is None:
        raise KeyError("Missing postprocess section in YAML")
    replaced = False
    for idx in range(start_index, len(lines)):
        line = lines[idx]
        stripped = line.lstrip()
        if not stripped:
            continue
        current_indent = len(line) - len(stripped)
        if current_indent <= postprocess_indent and not stripped.startswith("postprocess:"):
            break
        if stripped.startswith("#"):
            continue
        if stripped.startswith("threshold:"):
            original = line.rstrip("\n")
            if "#" in original:
                code_part, comment_part = original.split("#", 1)
                comment = "#" + comment_part.strip()
            else:
                code_part = original
                comment = ""
            indent = code_part[: len(code_part) - len(code_part.lstrip())]
            remainder = code_part[len(indent):]
            after_key = remainder[len("threshold:") :]
            spacing = after_key[: len(after_key) - len(after_key.lstrip())]
            value_str = format_float(float(new_value))
            new_line = f"{indent}threshold:{spacing}{value_str}"
            if comment:
                new_line = f"{new_line} {comment.strip()}"
            lines[idx] = new_line + "\n"
            replaced = True
            break
    if not replaced:
        raise KeyError("Missing threshold key under postprocess in YAML")
    with yaml_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def determine_output_path(explicit_path: str | None) -> Path:
    if explicit_path:
        output_path = Path(explicit_path)
        if output_path.parent:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    filename = f"threshold_{datetime.now().strftime('%Y%m%d')}.json"
    return reports_dir / filename


def main() -> None:
    args = parse_args()
    grid = args.grid or (0.01, 0.99, 0.01)
    start, end, step = map(float, grid)
    thresholds = build_threshold_grid(start, end, step)
    probs = pd.read_csv(args.oof)
    labels = pd.read_csv(args.labels)
    df = probs.merge(labels, on="filename", how="inner")
    y_true = df["label"].astype(int).values
    y_prob = df["prob"].astype(float).values
    results, best_thr, best_score = evaluate_thresholds(y_true, y_prob, thresholds)
    print(f"[Best] threshold={best_thr:.4f}, macro_f1={best_score:.5f}")
    grid_points = sample_grid_points(results, start, end, best_thr, best_score)
    output_path = determine_output_path(args.out)
    report = {
        "best_threshold": best_thr,
        "best_macro_f1": best_score,
        "grid_points": grid_points,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] saved to {output_path}")
    if args.write_yaml:
        update_threshold_yaml(Path(args.write_yaml), best_thr)
        print(
            f"[Update] postprocess.threshold -> {format_float(best_thr)} in {args.write_yaml}"
        )


if __name__ == "__main__":
    main()
