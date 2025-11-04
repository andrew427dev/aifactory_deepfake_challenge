# scripts/infer.py
import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment dependent
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:  # pragma: no cover - import success
    CV2_IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.loader import load_model_and_processor
from src.inference.aggregate_video import aggregate_probs
from src.utils.device import resolve_device, should_enable_amp
from src.utils.logging import get_logger

logger = get_logger("infer")

SUPPORTED_IMG = {".jpg", ".jpeg", ".png"}
SUPPORTED_VID = {".mp4", ".avi", ".mov", ".mkv"}


def require(cfg: Mapping[str, Any], path: str) -> Any:
    keys = path.split(".")
    current: Any = cfg
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"Missing required config key: {'.'.join(traversed)}")
        current = current[key]
    return current


def load_yaml(path: str) -> Mapping[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference entry point")
    parser.add_argument("--config", default="configs/infer.yaml", help="Path to inference config")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Data preprocessing config")
    parser.add_argument("--checkpoint", default=None, help="Optional local checkpoint to load")
    parser.add_argument("--save-probs", default=None, help="Optional path to save [filename, prob]")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    parser.add_argument("--dry-run", action="store_true", help="Print effective config and exit")
    return parser.parse_args()


def load_image_to_tensor(img_pil: Image.Image, processor) -> torch.Tensor:
    out = processor(img_pil)
    if isinstance(out, dict) and "pixel_values" in out:
        x = out["pixel_values"]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x  # BCHW
    raise RuntimeError("Unexpected processor output from image processor")


def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    """Try both call styles for timm and transformers models."""

    try:
        out = model(x)
        if hasattr(out, "logits"):
            return out.logits
        return out
    except TypeError:
        out = model(pixel_values=x)
        return out.logits if hasattr(out, "logits") else out


@torch.inference_mode()
def predict_pil(model, processor, device, amp: bool, img_pil: Image.Image) -> float:
    x = load_image_to_tensor(img_pil, processor).to(device)
    with torch.autocast(device_type=device.type, enabled=amp):
        logits = forward_logits(model, x)
    if logits.ndim == 2 and logits.shape[1] == 2:
        prob_fake = torch.softmax(logits.float(), dim=1)[:, 1]
    elif logits.ndim == 2 and logits.shape[1] == 1:
        prob_fake = torch.sigmoid(logits.float())[:, 0]
    else:  # single logit
        prob_fake = torch.sigmoid(logits.float()).reshape(-1)
    return float(prob_fake.squeeze().cpu())


def gather_inputs(manifest_path: Path | None, data_dir: Path | None) -> list[Path]:
    files: list[Path] = []
    if manifest_path and manifest_path.exists():
        df = pd.read_csv(manifest_path)
        col = "path" if "path" in df.columns else "filename"
        for v in df[col].tolist():
            p = Path(v)
            files.append(p if p.is_absolute() else (data_dir / v if data_dir else Path(v)))
    elif data_dir and data_dir.exists():
        files = [Path(p) for p in glob.glob(str(data_dir / "**/*"), recursive=True) if os.path.isfile(p)]
    else:
        raise FileNotFoundError("Neither valid manifest nor data_dir provided.")
    return sorted(files)


def decode_video_frames(path: Path, sample_fps: int, max_frames: int) -> list[Image.Image]:
    if cv2 is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "OpenCV is required for video decoding. Original import error: "
            f"{CV2_IMPORT_ERROR}"
        )
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", path)
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or sample_fps
    step = max(int(round(fps / sample_fps)), 1)
    frames: list[Image.Image] = []
    count = 0
    while len(frames) < max_frames:
        grabbed = cap.grab()
        if not grabbed:
            break
        if count % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        count += 1
    cap.release()
    return frames


def format_effective_config(effective: Mapping[str, Any]) -> str:
    dumped = yaml.safe_dump(effective, allow_unicode=True, sort_keys=False).strip()
    return dumped


def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)

    output_cfg = require(cfg, "output")
    out_submission = Path(require(output_cfg, "submission_path"))
    out_probs = Path(output_cfg.get("probs_path", "submission/probs.csv"))

    input_cfg = cfg.get("input", {})
    manifest_value = input_cfg.get("manifest") if input_cfg else None
    data_dir_value = input_cfg.get("data_dir") if input_cfg else None
    manifest_path = Path(manifest_value) if manifest_value else None
    data_dir = Path(data_dir_value) if data_dir_value else None

    runtime_cfg = require(cfg, "runtime")
    device_pref = str(require(runtime_cfg, "device"))
    amp_requested = bool(require(runtime_cfg, "amp"))
    batch_size = int(require(runtime_cfg, "batch_size"))
    num_workers = int(require(runtime_cfg, "num_workers"))
    pin_memory = bool(require(runtime_cfg, "pin_memory"))

    post_cfg = require(cfg, "postprocess")
    threshold_cfg = float(require(post_cfg, "threshold"))
    threshold = float(args.threshold) if args.threshold is not None else threshold_cfg

    sample_fps = int(require(data_cfg, "sample_fps"))
    max_frames = int(require(data_cfg, "max_frames"))
    img_size = int(require(data_cfg, "img_size"))
    normalize_cfg = require(data_cfg, "normalize")
    norm_mean = list(require(normalize_cfg, "mean"))
    norm_std = list(require(normalize_cfg, "std"))

    video_cfg = cfg.get("video_aggregate", {})
    vid_method = video_cfg.get("method", "mean")
    vid_topk = int(video_cfg.get("topk", 5))

    effective_cfg = {
        "output": {
            "submission_path": str(out_submission),
            "probs_path": str(out_probs),
        },
        "input": {
            "manifest": str(manifest_path) if manifest_path else None,
            "data_dir": str(data_dir) if data_dir else None,
        },
        "runtime": {
            "device": device_pref,
            "amp": amp_requested,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        },
        "postprocess": {"threshold": threshold},
        "data": {
            "sample_fps": sample_fps,
            "max_frames": max_frames,
            "img_size": img_size,
            "normalize": {"mean": norm_mean, "std": norm_std},
        },
    }

    logger.info("Effective configuration:\n%s", format_effective_config(effective_cfg))
    if args.dry_run:
        return

    device = resolve_device(device_pref)
    logger.info("Using device: %s", device.type)
    amp = should_enable_amp(device, amp_requested)
    logger.info("AMP enabled: %s", amp)

    model_cfg = load_yaml(args.model_config)
    model, processor, _ = load_model_and_processor(model_cfg)
    model.eval().to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint: %s", args.checkpoint)

    manifest_exists = manifest_path and manifest_path.exists()
    data_dir_exists = data_dir and data_dir.exists()
    files = gather_inputs(manifest_path if manifest_exists else None, data_dir if data_dir_exists else None)
    logger.info("Found %d inputs", len(files))

    rows, prob_rows = [], []

    for fpath in tqdm(files):
        ext = fpath.suffix.lower()
        fname = fpath.name

        if ext in SUPPORTED_IMG:
            img = Image.open(fpath).convert("RGB")
            prob = predict_pil(model, processor, device, amp, img)
            label = 1 if prob >= threshold else 0
            rows.append({"filename": fname, "label": int(label)})
            if args.save_probs or out_probs:
                prob_rows.append({"filename": fname, "prob": prob})
            continue

        if ext in SUPPORTED_VID:
            frame_dir = fpath.with_suffix("")
            frame_candidates = sorted(glob.glob(str(frame_dir / "*.jpg")))
            if frame_candidates:
                frame_candidates = frame_candidates[:max_frames]
                probs = []
                for fp in frame_candidates:
                    img = Image.open(fp).convert("RGB")
                    p = predict_pil(model, processor, device, amp, img)
                    probs.append(p)
            else:
                frames = decode_video_frames(fpath, sample_fps, max_frames)
                probs = [predict_pil(model, processor, device, amp, frame) for frame in frames]

            if len(probs) == 0:
                logger.warning("No frames to infer: %s", fpath)
                continue

            vid_label = aggregate_probs(probs, method=vid_method, topk=vid_topk, threshold=threshold)
            rows.append({"filename": fname, "label": int(vid_label)})
            if args.save_probs or out_probs:
                prob_rows.append({"filename": fname, "prob": float(sum(probs) / len(probs))})
            continue

        logger.warning("Skip unsupported file: %s", fpath)

    out_submission.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filename", "label"]).to_csv(out_submission, index=False)
    logger.info("Saved submission => %s", out_submission)

    if args.save_probs or len(prob_rows) > 0:
        target = Path(args.save_probs) if args.save_probs else out_probs
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prob_rows, columns=["filename", "prob"]).to_csv(target, index=False)
        logger.info("Saved probs => %s", target)


if __name__ == "__main__":
    main()
