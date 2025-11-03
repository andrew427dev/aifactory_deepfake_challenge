import os, yaml, torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("train")

def main(cfg_path="configs/train.yaml", model_cfg="configs/model.yaml"):
    set_seed( int(yaml.safe_load(open(cfg_path)).get("seed", 42)) )
    mcfg = yaml.safe_load(open(model_cfg))
    model_dir = os.path.dirname(mcfg.get("checkpoint","./model/model.safetensors")) or "./model"
    logger.info("Loading model from %s", model_dir)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    # 여기서부터 실제 Dataset/Dataloader/Optimizer 로직 덮어쓰기
    os.makedirs("experiments/exp001", exist_ok=True)
    save_path = "experiments/exp001/best.ckpt"
    torch.save(model.state_dict(), save_path)
    logger.info("Dummy training done. Saved checkpoint => %s", save_path)

if __name__ == "__main__":
    main()
