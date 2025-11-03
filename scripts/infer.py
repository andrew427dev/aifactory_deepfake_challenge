import os, glob, yaml, cv2, numpy as np, pandas as pd
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from src.utils.logging import get_logger
from src.inference.aggregate_video import aggregate_probs

logger = get_logger("infer")

@torch.no_grad()
def predict_image(model, processor, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img_rgb, return_tensors="pt")
    outputs = model(**{k: v for k,v in inputs.items()})
    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    # label2id: Realism=0, Deepfake=1 기준
    return float(probs[1])

def main(infer_cfg="configs/infer.yaml", model_cfg="configs/model.yaml", data_cfg="configs/data.yaml", checkpoint="experiments/exp001/best.ckpt"):
    icfg = yaml.safe_load(open(infer_cfg))
    mcfg = yaml.safe_load(open(model_cfg))
    dcfg = yaml.safe_load(open(data_cfg))

    submission_path = icfg["output"]["submission_path"]
    vid_method = icfg["video_aggregate"]["method"]
    vid_topk   = icfg["video_aggregate"]["topk"]
    threshold  = float(icfg["video_aggregate"]["threshold"])
    batch_size = icfg.get("batch_size", 64)

    model_dir = os.path.dirname(mcfg.get("checkpoint","./model/model.safetensors")) or "./model"
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    if os.path.exists(checkpoint):
        sd = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        logger.info("Loaded ckpt: %s", checkpoint)

    input_dir = dcfg["paths"]["input_dir"]
    files = sorted(glob.glob(os.path.join(input_dir, "*")))
    logger.info("Found %d files in %s", len(files), input_dir)

    rows = []
    for fpath in tqdm(files):
        fname = os.path.basename(fpath)
        ext = os.path.splitext(fname)[1].lower()
        if ext in [".jpg",".jpeg",".png"]:
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            p = predict_image(model, processor, img)
            label = 1 if p >= threshold else 0
        elif ext == ".mp4":
            # 균등 샘플링 추론(프레임 저장 없이 바로 캡처)
            cap = cv2.VideoCapture(fpath)
            if not cap.isOpened():
                logger.warning(f"Cannot open video: {fpath}"); label = 0
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                step = max(int(round(fps / 3)), 1)  # default 3 fps
                count, probs = 0, []
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    if count % step == 0:
                        probs.append(predict_image(model, processor, frame))
                    count += 1
                cap.release()
                label = aggregate_probs(probs, method=vid_method, topk=vid_topk, threshold=threshold)
        else:
            logger.warning(f"Skip unsupported file: {fpath}"); continue
        rows.append({"filename": fname, "label": int(label)})

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    pd.DataFrame(rows, columns=["filename","label"]).to_csv(submission_path, index=False)
    logger.info("Saved submission => %s", submission_path)

if __name__ == "__main__":
    main()
