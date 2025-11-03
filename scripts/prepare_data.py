# scripts/prepare_data.py
import os, glob, yaml, cv2, shutil, argparse
from pathlib import Path
from typing import Optional
from PIL import Image
from tqdm import tqdm
from src.utils.logging import get_logger

logger = get_logger("prepare")

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def extract_frames(video_path: str, out_dir: str, sample_fps: int = 3, max_frames: int = 24):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open: {video_path}"); return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(fps / sample_fps)), 1)
    count = saved = 0
    ensure_dir(out_dir)
    while True:
        ret = cap.grab()
        if not ret: break
        if count % step == 0:
            ret, frame = cap.retrieve()
            if not ret: break
            h, w = frame.shape[:2]
            if h == 0 or w == 0: break
            out_path = os.path.join(out_dir, f"{saved:04d}.jpg")
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved += 1
            if saved >= max_frames: break
        count += 1
    cap.release()
    return saved

def process(config_path: str, use_face: bool = False, detector: str = "mtcnn",
            size: int = 224, sample_fps: int = 3, max_frames: int = 24):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    input_dir = Path(cfg["paths"]["input_dir"])
    proc_dir  = Path(cfg["paths"]["processed_dir"])
    ensure_dir(proc_dir)

    # --- 얼굴 정렬기(옵션) 1회 생성 ---
    aligner = None
    if use_face:
        from src.preprocessing.face import FaceAligner
        aligner = FaceAligner(detector=detector, image_size=size)
        logger.info("FaceAligner initialized: %s", detector)

    files = sorted(glob.glob(str(input_dir / "**/*"), recursive=True))
    logger.info("Found %d files under %s", len(files), input_dir)

    def crop_and_save(img_path: str, out_path: str):
        if not use_face:
            shutil.copy2(img_path, out_path); return
        img = Image.open(img_path).convert("RGB")
        face = aligner(img) if aligner is not None else img.resize((size, size))
        face.save(out_path)

    for fpath in tqdm(files):
        if not os.path.isfile(fpath): continue
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
                logger.warning(f"No frames extracted: {fpath}")
                continue
            frame_files = sorted(glob.glob(str(vid_out / "*.jpg")))
            for fp in frame_files:
                crop_and_save(fp, fp)  # 제자리 덮어쓰기

        else:
            logger.warning(f"Skip unsupported file: {fpath}")

    logger.info("Done. processed assets are in %s", proc_dir)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--use-face-align", action="store_true", help="얼굴 정렬 사용")
    ap.add_argument("--detector", default="mtcnn", help="얼굴 검출기: mtcnn")
    ap.add_argument("--size", type=int, default=224, help="출력 이미지 크기")
    ap.add_argument("--fps", type=int, default=3, help="비디오 샘플링 FPS")
    ap.add_argument("--max-frames", type=int, default=24, help="비디오당 최대 프레임 수")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process(args.config, args.use_face_align, args.detector, args.size, args.fps, args.max_frames)
