# scripts/infer.py
import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference entry point")
    p.add_argument("--config", default="configs/infer.yaml", help="Path to inference config")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--checkpoint", default=None, help="Optional local checkpoint to load")
    p.add_argument("--save-probs", default=None, help="Optional path to save [filename, prob]")
    p.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    return p.parse_args()

def load_image_to_tensor(img_pil, processor):
    out = processor(img_pil)
    if isinstance(out, dict) and "pixel_values" in out:
        x = out["pixel_values"]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x  # BCHW
    raise RuntimeError("Unexpected processor output from image processor")

def forward_logits(model, x):
    """
    Try both call styles:
      - timm: model(x) -> Tensor
      - transformers: model(pixel_values=x) -> output.logits
    """
    try:
        out = model(x)
        if hasattr(out, "logits"):
            return out.logits
        return out
    except TypeError:
        out = model(pixel_values=x)
        return out.logits if hasattr(out, "logits") else out

@torch.inference_mode()
def predict_pil(model, processor, device, amp, img_pil):
    x = load_image_to_tensor(img_pil, processor).to(device)
    with torch.autocast(device_type=device.type, enabled=amp):
        logits = forward_logits(model, x)
    # shape handling
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
        # manifest에는 filename 또는 path 컬럼이 온다고 가정
        col = "path" if "path" in df.columns else "filename"
        for v in df[col].tolist():
            p = Path(v)
            files.append(p if p.is_absolute() else (data_dir / v if data_dir else Path(v)))
    elif data_dir and data_dir.exists():
        files = [Path(p) for p in glob.glob(str(data_dir / "**/*"), recursive=True) if os.path.isfile(p)]
    else:
        raise FileNotFoundError("Neither valid manifest nor data_dir provided.")
    return sorted(files)

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    # === config mapping ===
    out_submission = Path(cfg["output"]["submission_path"])
    out_probs      = Path(cfg["output"].get("probs_path", "submission/probs.csv"))
    manifest_path  = Path(cfg["input"].get("manifest", "")) if cfg.get("input") else None
    data_dir       = Path(cfg["input"].get("data_dir", "")) if cfg.get("input") else None

    runtime = cfg.get("runtime", {})
    device  = resolve_device(runtime.get("device", "auto"))
    amp     = should_enable_amp(device, runtime.get("amp", True))

    vid_cfg = cfg.get("video_aggregate", {})
    vid_method = vid_cfg.get("method", "mean")
    vid_topk   = int(vid_cfg.get("topk", 5))

    post_cfg = cfg.get("postprocess", {})
    threshold = float(args.threshold) if args.threshold is not None else float(post_cfg.get("threshold", 0.5))

    # === model & processor ===
    model, processor, _ = load_model_and_processor(model_cfg)
    model.eval().to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint: %s", args.checkpoint)

    files = gather_inputs(manifest_path if manifest_path and manifest_path.exists() else None,
                          data_dir if data_dir and data_dir.exists() else None)
    logger.info("Found %d inputs", len(files))

    rows, prob_rows = [], []

    for fpath in tqdm(files):
        ext = fpath.suffix.lower()
        fname = fpath.name

        # --- image ---
        if ext in SUPPORTED_IMG:
            img = Image.open(fpath).convert("RGB")
            prob = predict_pil(model, processor, device, amp, img)
            label = 1 if prob >= threshold else 0
            rows.append({"filename": fname, "label": int(label)})
            if args.save_probs or out_probs:
                prob_rows.append({"filename": fname, "prob": prob})
            continue

        # --- video ---
        if ext in SUPPORTED_VID:
            # 프레임이 미리 추출되어 있다면 같은 이름의 디렉터리에서 사용
            frame_dir = fpath.with_suffix("")
            frame_candidates = sorted(glob.glob(str(frame_dir / "*.jpg")))
            probs = []
            if len(frame_candidates) == 0:
                # 즉시 디코딩(느릴 수 있음, 24프레임 제한)
                cap = cv2.VideoCapture(str(fpath))
                ok, frame = cap.read()
                idx = 0
                while ok and idx < 24:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    p = predict_pil(model, processor, device, amp, img)
                    probs.append(p)
                    ok, frame = cap.read()
                    idx += 1
                cap.release()
            else:
                for fp in frame_candidates:
                    img = Image.open(fp).convert("RGB")
                    p = predict_pil(model, processor, device, amp, img)
                    probs.append(p)

            if len(probs) == 0:
                logger.warning("No frames to infer: %s", fpath)
                continue

            vid_label = aggregate_probs(probs, method=vid_method, topk=vid_topk, threshold=threshold)
            rows.append({"filename": fname, "label": int(vid_label)})
            if args.save_probs or out_probs:
                prob_rows.append({"filename": fname, "prob": float(sum(probs) / len(probs))})
            continue

        logger.warning("Skip unsupported file: %s", fpath)

    # --- save outputs ---
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
