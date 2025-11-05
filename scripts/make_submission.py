import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from src.submission.schema_check import validate_submission


PATH_CANDIDATES = [
    "filename",
    "video_id",
    "filepath",
    "file_path",
    "path",
    "image_path",
    "frame_path",
]


def load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def detect_column(columns: Iterable[str]) -> str:
    for name in PATH_CANDIDATES:
        if name in columns:
            return name
    raise KeyError(
        f"Manifest must contain one of columns {PATH_CANDIDATES}; found {list(columns)}"
    )


def manifest_ids(manifest_path: Path) -> set[str]:
    frame = pd.read_csv(manifest_path)
    if frame.empty:
        raise ValueError(f"Manifest {manifest_path} is empty")
    column = detect_column(frame.columns)
    values = frame[column].astype(str)
    return {Path(v).name for v in values}


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate submission outputs and sanity-check scores")
    ap.add_argument("--config", default="configs/infer.yaml", help="Path to inference config")
    ap.add_argument("--o", "--output", dest="out", default="submission/submission.csv")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    output_cfg = cfg.get("output", {})
    submission_path = Path(args.out)

    probs_path = output_cfg.get("probs_path")
    if not probs_path:
        raise ValueError("Config must define output.probs_path for submission checks.")
    probs_path = Path(probs_path)
    if not probs_path.exists():
        raise FileNotFoundError(f"Probabilities CSV not found: {probs_path}")

    probs_df = pd.read_csv(probs_path)
    required_cols = {"video_id", "score"}
    if not required_cols.issubset(probs_df.columns):
        raise ValueError(
            f"Probabilities file must contain columns {sorted(required_cols)}; got {list(probs_df.columns)}"
        )
    if probs_df["video_id"].duplicated().any():
        raise ValueError("Duplicate video_id entries detected in probabilities CSV.")
    if probs_df["score"].isna().any():
        raise ValueError("Probabilities CSV contains NaN scores.")

    expected_ids: set[str] = set()
    input_cfg = cfg.get("input", {})
    manifest_value = input_cfg.get("manifest") if input_cfg else None
    if manifest_value:
        manifest_path = Path(manifest_value)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        expected_ids = manifest_ids(manifest_path)
        missing_from_probs = expected_ids - set(probs_df["video_id"].astype(str))
        if missing_from_probs:
            sample = ", ".join(sorted(missing_from_probs)[:5])
            raise ValueError(
                f"Probabilities CSV missing {len(missing_from_probs)} ids from manifest. Sample: {sample}"
            )

    validate_submission(str(submission_path))
    submission_df = pd.read_csv(submission_path)

    if expected_ids:
        missing_from_submission = expected_ids - set(submission_df["filename"].astype(str))
        if missing_from_submission:
            sample = ", ".join(sorted(missing_from_submission)[:5])
            raise ValueError(
                f"Submission CSV missing {len(missing_from_submission)} ids from manifest. Sample: {sample}"
            )

    postprocess_cfg = cfg.get("postprocess", {})
    threshold = postprocess_cfg.get("threshold")
    total = len(submission_df)
    positives = int((submission_df["label"] == 1).sum())
    print(
        f"Submission summary | total={total} | positives={positives} | threshold={threshold}"
    )


if __name__ == "__main__":
    main()
