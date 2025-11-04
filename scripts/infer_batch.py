# scripts/infer_batch.py
import argparse, os, sys, glob
from pathlib import Path
import yaml, torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.loader import load_model_and_processor
from src.utils.device import resolve_device, should_enable_amp
from src.utils.logging import get_logger
from src.inference.aggregate_video import aggregate_probs
from src.inference.postprocess import binarize

logger = get_logger("infer_batch")

SUPPORTED_IMG = {".jpg", ".jpeg", ".png"}
SUPPORTED_VID = {".mp4", ".avi", ".mov", ".mkv"}

def gather_files(manifest, data_dir):
    files = []
    if manifest and Path(manifest).exists():
        df = pd.read_csv(manifest)
        col = "path" if "path" in df.columns else "filename"
        for v in df[col].tolist():
            p = Path(v)
            files.append(p if p.is_absolute() else (Path(data_dir)/v if data_dir else Path(v)))
    elif data_dir and Path(data_dir).exists():
        files = [Path(p) for p in glob.glob(str(Path(data_dir)/"**/*"), recursive=True) if os.path.isfile(p)]
    else:
        raise FileNotFoundError("No valid manifest or data_dir")
    return sorted(files)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/infer.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    return ap.parse_args()

def make_pil_loader(processor):
    """processor를 활용해 PIL.Image -> 텐서로 변환하는 함수 반환"""
    def pil_to_tensor(pil):
        out = processor(pil)
        x = out["pixel_values"] if isinstance(out, dict) else out
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x
    return pil_to_tensor

def parallel_load(paths, pil_to_tensor, num_workers: int):
    """paths(list[str|Path])를 스레드풀로 병렬 로딩하여 텐서 리스트 반환"""
    results = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        futs = {ex.submit(_load_one, i, p, pil_to_tensor): i for i, p in enumerate(paths)}
        for fut in as_completed(futs):
            idx, tensor = fut.result()
            results[idx] = tensor
    return results

def _load_one(idx, path, pil_to_tensor):
    img = Image.open(path).convert("RGB")
    return idx, pil_to_tensor(img)

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    mcfg = yaml.safe_load(open(args.model_config))

    output_cfg = cfg.get("output", {})
    out_submission = Path(output_cfg["submission_path"])
    probs_value = output_cfg.get("probs_path")
    out_probs = Path(probs_value) if probs_value else None
    manifest       = cfg["input"].get("manifest")
    data_dir       = cfg["input"].get("data_dir")

    runtime = cfg.get("runtime", {})
    device  = resolve_device(runtime.get("device", "auto"))
    amp     = should_enable_amp(device, runtime.get("amp", True))
    bsz     = int(runtime.get("batch_size", 64))
    nworkers= int(runtime.get("num_workers", 0))  # 0이면 직렬

    post_cfg = cfg.get("postprocess", {})
    threshold = float(args.threshold) if args.threshold is not None else float(post_cfg.get("threshold", 0.5))

    vid_cfg = cfg.get("video_aggregate", {})
    vid_method = vid_cfg.get("method", "mean")
    vid_topk   = int(vid_cfg.get("topk", 5))

    model, processor, _ = load_model_and_processor(mcfg)
    model.eval().to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu")
        if "state_dict" in state: state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint: %s", args.checkpoint)

    files = gather_files(manifest, data_dir)
    logger.info("Found %d files", len(files))

    images = []
    videos = {}
    for f in files:
        ext = f.suffix.lower()
        if ext in SUPPORTED_IMG:
            images.append(f)
        elif ext in SUPPORTED_VID:
            frame_dir = f.with_suffix("")
            frames = sorted(glob.glob(str(frame_dir/"*.jpg")))
            if not frames:
                continue
            videos[str(f)] = frames

    rows, score_rows = [], []
    pil_to_tensor = make_pil_loader(processor)

    def run_model_on_batch(x):
        with torch.autocast(device_type=device.type, enabled=amp):
            try:
                logits = model(x)
                logits = logits.logits if hasattr(logits, "logits") else logits
            except TypeError:
                out = model(pixel_values=x)
                logits = out.logits if hasattr(out, "logits") else out
        if logits.ndim == 2 and logits.shape[1] == 2:
            probs = torch.softmax(logits.float(), dim=1)[:,1]
        elif logits.ndim == 2 and logits.shape[1] == 1:
            probs = torch.sigmoid(logits.float())[:,0]
        else:
            probs = torch.sigmoid(logits.float()).reshape(-1)
        return probs.detach().cpu()

    # ----- 이미지 일괄 추론 -----
    i = 0
    while i < len(images):
        chunk = images[i:i+bsz]
        # 병렬 로딩
        if nworkers > 0:
            tensors = parallel_load(chunk, pil_to_tensor, nworkers)
        else:
            tensors = [pil_to_tensor(Image.open(p).convert("RGB")) for p in chunk]
        x = torch.cat(tensors, dim=0).to(device, non_blocking=True)
        probs = run_model_on_batch(x).tolist()
        for p, pr in zip(chunk, probs):
            label = binarize(pr, threshold)
            rows.append({"filename": p.name, "label": int(label)})
            if out_probs is not None:
                score_rows.append({"video_id": p.name, "score": float(pr)})
        i += bsz

    # ----- 비디오(프레임) 일괄 추론 -----
    for vpath, frame_list in tqdm(videos.items(), desc="videos"):
        probs = []
        j = 0
        while j < len(frame_list):
            chunk = frame_list[j:j+bsz]
            if nworkers > 0:
                tensors = parallel_load(chunk, pil_to_tensor, nworkers)
            else:
                tensors = [pil_to_tensor(Image.open(fp).convert("RGB")) for fp in chunk]
            x = torch.cat(tensors, dim=0).to(device, non_blocking=True)
            chunk_probs = run_model_on_batch(x).tolist()
            probs.extend(chunk_probs)
            j += bsz
        score = aggregate_probs(probs, method=vid_method, topk=vid_topk)
        label = binarize(score, threshold)
        rows.append({"filename": Path(vpath).name, "label": int(label)})
        if out_probs is not None:
            score_rows.append({"video_id": Path(vpath).name, "score": float(score)})

    out_submission.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filename","label"]).to_csv(out_submission, index=False)

    if out_probs is not None:
        out_probs.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(score_rows, columns=["video_id", "score"]).to_csv(out_probs, index=False)
        logger.info("Saved scores => %s", out_probs)

    logger.info("Saved submission => %s", out_submission)

if __name__ == "__main__":
    main()
