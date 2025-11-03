import os, yaml, torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from src.utils.logging import get_logger
from src.evaluation.metrics import macro_f1

logger = get_logger("validate")

def main(cfg_path="configs/train.yaml", checkpoint="experiments/exp001/best.ckpt", model_cfg="configs/model.yaml"):
    mcfg = yaml.safe_load(open(model_cfg))
    model_dir = os.path.dirname(mcfg.get("checkpoint","./model/model.safetensors")) or "./model"
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    if os.path.exists(checkpoint):
        sd = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        logger.info("Loaded checkpoint: %s", checkpoint)
    else:
        logger.warning("Checkpoint not found. Using base weights.")

    # 더미 검증: 빈 데이터 → macro_f1 계산 스킵
    logger.info("No validation dataset wired yet. Skipping metric calc.")
    print("macro_f1: N/A (wire dataset to compute)")
    
if __name__ == "__main__":
    main()
