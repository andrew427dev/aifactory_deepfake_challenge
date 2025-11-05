# scripts/prepare_data.py
import argparse
import glob
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
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


def _compute_frame_indices(
    fps: float, frame_count: int | None, sample_fps: int, max_frames: int
) -> list[int]:
    if fps <= 0:
        fps = float(sample_fps)
    indices: list[int] = []
    interval = fps / float(sample_fps)
    for idx in range(max_frames):
        frame_idx = int(round(idx * interval))
        if frame_count is not None and frame_idx >= frame_count:
            break
        if indices and frame_idx <= indices[-1]:
            frame_idx = indices[-1] + 1
            if frame_count is not None and frame_idx >= frame_count:
                break
        indices.append(frame_idx)
    return indices


def extract_frames(
    video_path: Path,
    out_dir: Path,
    sample_fps: int,
    max_frames: int,
    aligner: Any | None,
) -> int:
    if cv2 is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "OpenCV is required to extract frames. Original import error: "
            f"{CV2_IMPORT_ERROR}"
        )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return -1

    raw_fps = cap.get(cv2.CAP_PROP_FPS) or float(sample_fps)
    frame_count_val = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = frame_count_val if frame_count_val > 0 else None
    indices = _compute_frame_indices(raw_fps, frame_count, sample_fps, max_frames)

    if not indices:
        logger.warning("No frame indices selected for %s", video_path)
        cap.release()
        return 0

    ensure_dir(out_dir)
    saved = 0
    target_iter = iter(indices)
    target = next(target_iter, None)
    current_idx = 0

    while target is not None:
        ret, frame = cap.read()
        if not ret:
            logger.warning(
                "Video ended before reaching target frame %d in %s", target, video_path
            )
            break
        if current_idx == target:
            out_path = out_dir / f"{saved:04d}.jpg"
            if aligner is not None:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                face = aligner(image)
                face.save(out_path)
            else:
                cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved += 1
            target = next(target_iter, None)
        current_idx += 1

    cap.release()
    return saved


def process(
    config_path: str,
    mode: str,
    use_face: bool = False,
    detector: str = "mtcnn",
) -> None:
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

    stats = defaultdict(int)

    def process_image_file(image_path: Path, rel_path: Path) -> None:
        if len(rel_path.parts) > 1:
            split = rel_path.parts[0]
            remainder = Path(*rel_path.parts[1:])
        else:
            split = "default"
            remainder = rel_path

        components = list(remainder.parts[:-1]) + [remainder.stem]
        sample_name = "__".join([comp for comp in components if comp]) or remainder.stem
        sample_dir = Path(sample_name)
        counter_key = (split, sample_name)
        idx = image_counters[counter_key]
        image_counters[counter_key] += 1

        out_dir = proc_dir / split / sample_dir
        ensure_dir(out_dir)
        out_path = out_dir / f"{idx:04d}.jpg"

        try:
            img = Image.open(image_path).convert("RGB")
            if aligner is not None:
                result = aligner(img)
            else:
                result = img.resize((size, size), Image.BILINEAR)
            result.save(out_path)
            stats["processed"] += 1
        except Exception as exc:  # pragma: no cover - file-specific errors
            logger.warning("Failed to process image %s: %s", image_path, exc)
            stats["errors"] += 1

    image_counters: defaultdict[tuple[str, str], int] = defaultdict(int)

    for fpath in tqdm(files):
        path_obj = Path(fpath)
        if not path_obj.is_file():
            continue
        rel = path_obj.relative_to(input_dir)
        ext = path_obj.suffix.lower()

        if mode == "image" and ext in {".jpg", ".jpeg", ".png"}:
            process_image_file(path_obj, rel)
            continue

        if mode == "video" and ext in {".mp4", ".mov", ".avi"}:
            out_dir = proc_dir / rel.with_suffix("")
            try:
                extracted = extract_frames(
                    path_obj,
                    out_dir,
                    sample_fps,
                    max_frames,
                    aligner,
                )
            except RuntimeError:
                raise
            except Exception as exc:  # pragma: no cover - codec specific
                logger.warning("Failed to extract %s: %s", path_obj, exc)
                stats["errors"] += 1
                continue

            if extracted < 0:
                stats["errors"] += 1
                continue

            if extracted == 0:
                logger.warning("No frames extracted: %s", path_obj)
                stats["skipped"] += 1
                continue

            stats["processed"] += 1
            continue

        # Unsupported or mismatched mode
        if ext in {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi"}:
            stats["skipped"] += 1
            logger.warning("Skip %s for mode %s", path_obj, mode)
        else:
            logger.warning("Skip unsupported file: %s", path_obj)
            stats["skipped"] += 1

    logger.info(
        "Summary - processed: %d, skipped: %d, errors: %d",
        stats.get("processed", 0),
        stats.get("skipped", 0),
        stats.get("errors", 0),
    )
    logger.info("Processed assets are in %s", proc_dir)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument(
        "--mode",
        default="video",
        choices=["video", "image"],
        help="입력 자산 유형 (video 또는 image)",
    )
    ap.add_argument("--use-face-align", action="store_true", help="얼굴 정렬 사용")
    ap.add_argument("--detector", default="mtcnn", help="얼굴 검출기: mtcnn")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    reports_dir = ROOT / "reports"
    ensure_dir(reports_dir)
    log_path = reports_dir / f"prepare_{datetime.now().strftime('%Y%m%d')}.log"

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info("Log file: %s", log_path)

    try:
        process(args.config, args.mode, args.use_face_align, args.detector)
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()
