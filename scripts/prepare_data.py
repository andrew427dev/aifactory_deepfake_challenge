# scripts/prepare_data.py
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

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

from src.utils.logging import get_logger

logger = get_logger("prepare")


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


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


def extract_frames(video_path: str, out_dir: str, sample_fps: int, max_frames: int) -> int:
    if cv2 is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "OpenCV is required to extract frames. Original import error: "
            f"{CV2_IMPORT_ERROR}"
        )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open: %s", video_path)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or sample_fps
    step = max(int(round(fps / sample_fps)), 1)
    count = saved = 0
    ensure_dir(out_dir)
    while True:
        ret = cap.grab()
        if not ret:
            break
        if count % step == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                break
            out_path = os.path.join(out_dir, f"{saved:04d}.jpg")
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved += 1
            if saved >= max_frames:
                break
        count += 1
    cap.release()
    return saved


def process(config_path: str, use_face: bool = False, detector: str = "mtcnn") -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    size = int(require(cfg, "img_size"))
    sample_fps = int(require(cfg, "sample_fps"))
    max_frames = int(require(cfg, "max_frames"))

    paths_cfg = require(cfg, "paths")
    input_dir = Path(require(paths_cfg, "input_dir"))
    proc_dir = Path(require(paths_cfg, "processed_dir"))
    ensure_dir(proc_dir)

    # --- 얼굴 정렬기(옵션) 1회 생성 ---
    aligner = None
    if use_face:
        from src.preprocessing.face import FaceAligner

        aligner = FaceAligner(detector=detector, image_size=size)
        logger.info("FaceAligner initialized: %s", detector)

    files = sorted(glob.glob(str(input_dir / "**/*"), recursive=True))
    logger.info("Found %d files under %s", len(files), input_dir)

    def crop_and_save(img_path: str, out_path: str) -> None:
        if not use_face:
            shutil.copy2(img_path, out_path)
            return
        img = Image.open(img_path).convert("RGB")
        face = aligner(img) if aligner is not None else img.resize((size, size))
        face.save(out_path)

    for fpath in tqdm(files):
        if not os.path.isfile(fpath):
            continue
        rel = os.path.relpath(fpath, input_dir)
        base = os.path.basename(fpath)
        ext = os.path.splitext(base)[1].lower()

        if ext in [".jpg", ".jpeg", ".png"]:
            out_img_dir = proc_dir / os.path.dirname(rel)
            ensure_dir(out_img_dir)
            out_path = out_img_dir / base
            crop_and_save(fpath, str(out_path))

        elif ext == ".mp4":
            vid_out = proc_dir / os.path.splitext(rel)[0]
            n = extract_frames(fpath, str(vid_out), sample_fps, max_frames)
            if n == 0:
                logger.warning("No frames extracted: %s", fpath)
                continue
            frame_files = sorted(glob.glob(str(vid_out / "*.jpg")))
            for fp in frame_files:
                crop_and_save(fp, fp)  # 제자리 덮어쓰기

        else:
            logger.warning("Skip unsupported file: %s", fpath)

    logger.info("Done. processed assets are in %s", proc_dir)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--use-face-align", action="store_true", help="얼굴 정렬 사용")
    ap.add_argument("--detector", default="mtcnn", help="얼굴 검출기: mtcnn")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(args.config, args.use_face_align, args.detector)
