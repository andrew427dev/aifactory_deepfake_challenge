import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.loader import load_model_and_processor, resolve_model_source
from src.training.trainer import Trainer
from src.utils.device import resolve_device, should_enable_amp
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training entry point")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training config")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="Override runtime.device")
    return parser.parse_args()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)

    set_seed(int(cfg.get("seed", 42)))

    runtime_cfg = cfg.get("runtime", {})
    device_pref = args.device if args.device is not None else runtime_cfg.get("device", "auto")
    device = resolve_device(device_pref)
    logger.info("Using device: %s", device.type)

    amp_requested = runtime_cfg.get("amp", True)
    amp_enabled = should_enable_amp(device, amp_requested)
    logger.info("AMP enabled: %s", amp_enabled)

    model_source = resolve_model_source(model_cfg)
    logger.info("Loading model resources from %s", model_source)

    model, _, is_fallback = load_model_and_processor(model_cfg)
    if is_fallback:
        logger.warning("Using fallback simple model; pretrained weights were unavailable.")

    trainer = Trainer(model=model, device=device, amp=amp_enabled, logger=logger)
    trainer.configure_optimizer(cfg.get("optimizer", {}))

    trainer_cfg = cfg.get("trainer", {})
    dataloader_cfg = cfg.get("dataloader", {})
    dummy_steps = max(1, int(trainer_cfg.get("dummy_steps", 1)))
    epochs = max(1, int(trainer_cfg.get("epochs", 1)))
    batch_size = max(1, int(dataloader_cfg.get("train", {}).get("batch_size", 4)))

    trainer.fit_dummy(
        epochs=epochs,
        steps_per_epoch=dummy_steps,
        batch_size=batch_size,
    )

    output_dir = Path(cfg.get("paths", {}).get("output_dir", "experiments/exp001"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "best.ckpt"
    trainer.save_checkpoint(save_path)
    logger.info("Dummy training done. Saved checkpoint => %s", save_path)


if __name__ == "__main__":
    main()
