import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.loader import load_model_and_processor, resolve_model_source
from src.inference.aggregate_video import aggregate_probs
from src.utils.device import resolve_device, should_enable_amp
from src.utils.logging import get_logger

logger = get_logger("infer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference entry point")
    parser.add_argument("--config", default="configs/infer.yaml", help="Path to inference config")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Path to data config")
    parser.add_argument("--checkpoint", default="experiments/exp001/best.ckpt", help="Model checkpoint path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="Override runtime.device")
    parser.add_argument("--disable-amp", action="store_true", help="Force disable AMP regardless of config")
    return parser.parse_args()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@torch.no_grad()
def predict_image(model, processor, img_bgr, device, amp_enabled):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    batch = (
        processor(images=img_rgb, return_tensors="pt")
        if processor
        else {"pixel_values": torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0}
    )
    if hasattr(batch, "to"):
        inputs = batch.to(device)
    else:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        outputs = model(**{k: v for k, v in inputs.items()})
    probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()
    return float(probs[1])


def main():
    args = parse_args()

    infer_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config)

    runtime_cfg = infer_cfg.get("runtime", {})
    device_pref = args.device if args.device is not None else runtime_cfg.get("device", "auto")
    device = resolve_device(device_pref)
    logger.info("Using device: %s", device.type)

    amp_requested = runtime_cfg.get("amp", True)
    if args.disable_amp:
        amp_requested = False
    amp_enabled = should_enable_amp(device, amp_requested)
    logger.info("AMP enabled: %s", amp_enabled)

    model_source = resolve_model_source(model_cfg)
    logger.info("Loading model resources from %s", model_source)
    model, processor, is_fallback = load_model_and_processor(model_cfg)
    if is_fallback:
        logger.warning("Using fallback simple model; pretrained weights were unavailable.")
    model = model.to(device)
    model.eval()

    checkpoint = args.checkpoint
    if os.path.exists(checkpoint):
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded ckpt: %s", checkpoint)
    else:
        logger.warning("Checkpoint not found at %s; using base weights.", checkpoint)

    submission_path = infer_cfg["output"]["submission_path"]
    vid_method = infer_cfg["video_aggregate"]["method"]
    vid_topk = infer_cfg["video_aggregate"]["topk"]
    threshold = float(infer_cfg["video_aggregate"]["threshold"])

    input_dir = data_cfg["paths"]["input_dir"]
    files = sorted(glob.glob(os.path.join(input_dir, "*")))
    logger.info("Found %d files in %s", len(files), input_dir)

    rows = []
    for fpath in tqdm(files):
        fname = os.path.basename(fpath)
        ext = os.path.splitext(fname)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            p = predict_image(model, processor, img, device, amp_enabled)
            label = 1 if p >= threshold else 0
        elif ext == ".mp4":
            cap = cv2.VideoCapture(fpath)
            if not cap.isOpened():
                logger.warning("Cannot open video: %s", fpath)
                label = 0
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                step = max(int(round(fps / 3)), 1)
                count, probs = 0, []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if count % step == 0:
                        probs.append(predict_image(model, processor, frame, device, amp_enabled))
                    count += 1
                cap.release()
                label = aggregate_probs(probs, method=vid_method, topk=vid_topk, threshold=threshold)
        else:
            logger.warning("Skip unsupported file: %s", fpath)
            continue
        rows.append({"filename": fname, "label": int(label)})

    submission_path = Path(submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filename", "label"]).to_csv(submission_path, index=False)
    logger.info("Saved submission => %s", submission_path)


if __name__ == "__main__":
    main()
