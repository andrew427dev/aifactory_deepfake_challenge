import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import build_dataloader, load_yaml  # noqa: E402
from src.evaluation.validator import Validator  # noqa: E402
from src.models.loader import load_model_and_processor, resolve_model_source  # noqa: E402
from src.utils.device import resolve_device, should_enable_amp  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("validate")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation entry point")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training config")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", default="experiments/exp001/best.ckpt", help="Checkpoint to evaluate")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="Override runtime.device")
    parser.add_argument("--disable-amp", action="store_true", help="Force disable AMP regardless of config")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Shared data configuration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config) or {}
    model_cfg = load_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config) or {}

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    runtime_cfg = require(cfg, "runtime")
    device_pref = args.device if args.device is not None else str(require(runtime_cfg, "device"))
    device = resolve_device(device_pref)
    logger.info("Using device: %s", device.type)

    amp_requested = bool(require(runtime_cfg, "amp"))
    if args.disable_amp:
        amp_requested = False
    amp_enabled = should_enable_amp(device, amp_requested)
    logger.info("AMP enabled: %s", amp_enabled)

    model_source = resolve_model_source(model_cfg)
    logger.info("Loading model resources from %s", model_source)
    model, _processor, is_fallback = load_model_and_processor(model_cfg)
    if is_fallback:
        logger.warning("Using fallback simple model; pretrained weights were unavailable.")

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded checkpoint: %s", checkpoint_path)
    else:
        logger.warning("Checkpoint not found at %s; using base weights.", checkpoint_path)

    trainer_cfg = cfg.get("trainer", {})
    dataloader_cfg = cfg.get("dataloader", {})
    train_data_cfg = cfg.get("data", {})
    runtime_paths = cfg.get("paths", {})

    data_format = str(train_data_cfg.get("format", "image")).lower()
    normalize_cfg = require(data_cfg, "normalize")
    image_size = int(require(data_cfg, "img_size"))
    mean = list(require(normalize_cfg, "mean"))
    std = list(require(normalize_cfg, "std"))
    max_frames = int(require(data_cfg, "max_frames"))
    path_column = train_data_cfg.get("path_column")
    label_column = train_data_cfg.get("label_column")

    val_manifest = runtime_paths.get("val_manifest")
    fallback_val = int(trainer_cfg.get("fallback_val_size", 64))
    model_config = getattr(model, "config", None)
    num_labels = getattr(model_config, "num_labels", 2) if model_config is not None else 2

    val_loader = build_dataloader(
        manifest=val_manifest,
        loader_cfg=dataloader_cfg.get("val", {}),
        fallback_length=fallback_val,
        dataset_kind=data_format,
        image_size=image_size,
        mean=mean,
        std=std,
        max_frames=max_frames,
        num_labels=num_labels,
        seed=seed,
        path_column=path_column,
        label_column=label_column,
        shuffle_override=False,
    )

    validator = Validator(model=model, device=device, amp=amp_enabled, logger=logger)

    epochs = max(1, int(trainer_cfg.get("epochs", trainer_cfg.get("max_epochs", 1))))
    best_result = None

    for epoch in range(1, epochs + 1):
        result = validator.evaluate(val_loader)
        logger.info(
            "Epoch %d validation | loss=%.4f | macro_f1=%.4f",
            epoch,
            result.loss,
            result.macro_f1,
        )
        print(f"Epoch {epoch}: loss={result.loss:.4f} macro_f1={result.macro_f1:.4f}")
        if best_result is None or result.macro_f1 > best_result.macro_f1:
            best_result = result

    if best_result is not None:
        logger.info("Best Macro F1 across epochs: %.4f", best_result.macro_f1)


if __name__ == "__main__":
    main()
