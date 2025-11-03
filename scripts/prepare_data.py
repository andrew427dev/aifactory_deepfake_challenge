import os, glob, yaml, cv2, shutil
from tqdm import tqdm
from src.utils.logging import get_logger

logger = get_logger("prepare")

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def extract_frames(video_path, out_dir, sample_fps=3, max_frames=24):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        logger.warning(f"Cannot open: {video_path}"); return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(fps / sample_fps)), 1)
    count = saved = 0
    ensure_dir(out_dir)
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % step == 0:
            save_path = os.path.join(out_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(save_path, frame)
            saved += 1
            if saved >= max_frames: break
        count += 1
    cap.release()
    return saved

def main(cfg_path="configs/data.yaml"):
    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    in_dir  = cfg["paths"]["input_dir"]
    proc_dir= cfg["paths"]["processed_dir"]
    img_size= cfg.get("image_size", 224)
    vcfg    = cfg.get("video", {})
    sample_fps = vcfg.get("sample_fps", 3)
    max_frames = vcfg.get("max_frames", 24)

    ensure_dir(proc_dir)
    files = sorted(glob.glob(os.path.join(in_dir, "*")))
    logger.info(f"Found {len(files)} files in {in_dir}")

    for fpath in tqdm(files):
        base = os.path.basename(fpath)
        ext = os.path.splitext(base)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            # 그냥 복사 (추후 얼굴/정규화 파이프라인 여지)
            shutil.copy2(fpath, os.path.join(proc_dir, base))
        elif ext == ".mp4":
            vid_out = os.path.join(proc_dir, os.path.splitext(base)[0])
            n = extract_frames(fpath, vid_out, sample_fps, max_frames)
            if n == 0: logger.warning(f"No frames extracted: {fpath}")
        else:
            logger.warning(f"Skip unsupported file: {fpath}")
    logger.info("Done. processed assets are in %s", proc_dir)

if __name__ == "__main__":
    main()
