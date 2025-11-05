import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import build_dataloader, load_yaml  # noqa: E402
from src.evaluation.validator import Validator  # noqa: E402
from src.models.loader import load_model_and_processor, resolve_model_source  # noqa: E402
from src.utils.device import enable_perf_flags, resolve_device, should_enable_amp  # noqa: E402
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
    parser.add_argument("--smoke-test", action="store_true", help="Run a lightweight model/processor smoke test and exit")
    parser.add_argument("--fold", type=int, default=None, help="Optional fold index for CV logging")
    parser.add_argument(
        "--save-oof",
        default="submission/oof.csv",
        help="Path to save out-of-fold predictions [filename, prob, label_true]",
    )
    return parser.parse_args()


def _resolve_processor_size(processor: Any) -> int:
    size_attr = getattr(processor, "size", None)
    if isinstance(size_attr, dict):
        for key in ("shortest_edge", "height", "width"):
            if key in size_attr:
                return int(size_attr[key])
        if size_attr:
            first = next(iter(size_attr.values()))
            return int(first)
    if isinstance(size_attr, (list, tuple)) and size_attr:
        return int(size_attr[0])
    if size_attr is not None:
        try:
            return int(size_attr)
        except (TypeError, ValueError):
            pass
    crop = getattr(processor, "crop_size", None)
    if isinstance(crop, dict) and crop:
        first = next(iter(crop.values()))
        return int(first)
    if crop is not None:
        try:
            return int(crop)
        except (TypeError, ValueError):
            pass
    return 224


def _forward_logits(model: torch.nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    try:
        outputs = model(pixel_values=pixel_values)
    except TypeError:
        outputs = model(pixel_values)
    return outputs.logits if hasattr(outputs, "logits") else outputs


def run_model_smoke_test(model: torch.nn.Module, processor: Any, device: torch.device) -> None:
    model = model.to(device)
    model.eval()
    size = _resolve_processor_size(processor)

    img1 = Image.new("RGB", (size, size), color=(255, 0, 0))
    import numpy as np  # local import to avoid hard dependency at module load

    img2 = np.zeros((size, size, 3), dtype=np.uint8)
    img2[..., 1] = 128

    batch = processor([img1, img2], return_tensors="pt")
    pixel_values = batch.get("pixel_values") if isinstance(batch, Mapping) else batch
    if pixel_values is None:
        raise RuntimeError("Processor output does not contain 'pixel_values'")
    if not isinstance(pixel_values, torch.Tensor):
        pixel_values = torch.tensor(pixel_values)
    pixel_values = pixel_values.to(device)

    with torch.inference_mode():
        logits = _forward_logits(model, pixel_values)

    if logits.ndim < 2 or logits.shape[0] != 2:
        raise RuntimeError(f"Unexpected smoke test logits shape: {tuple(logits.shape)}")
    if torch.isnan(logits).any():
        raise RuntimeError("Smoke test produced NaN logits")
    logger.info("Smoke test succeeded | logits shape=%s", tuple(logits.shape))


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

    channels_last = bool(runtime_cfg.get("channels_last", False))
    cudnn_benchmark = bool(runtime_cfg.get("cudnn_benchmark", False))
    enable_perf_flags(channels_last=channels_last, cudnn_benchmark=cudnn_benchmark)

    model_source = resolve_model_source(model_cfg)
    logger.info("Loading model resources from %s", model_source)
    model, processor, is_fallback = load_model_and_processor(model_cfg)
    if is_fallback:
        logger.warning("Using fallback simple model; pretrained weights were unavailable.")

    model = model.to(device)

    if args.smoke_test:
        run_model_smoke_test(model, processor, device)
        return

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

    result, details = validator.evaluate(val_loader, return_details=True)
    logger.info("Validation completed | loss=%.4f | macro_f1=%.4f", result.loss, result.macro_f1)
    print(f"Validation: loss={result.loss:.4f} macro_f1={result.macro_f1:.4f}")

    filenames = details.get("filenames", [])
    probs = details.get("probs", [])
    labels_true = details.get("labels", [])

    if len(filenames) != len(probs) or len(probs) != len(labels_true):
        raise RuntimeError(
            "Mismatch between filenames, probabilities, and labels lengths for OOF export."
        )

    oof_path = Path(args.save_oof)
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame(
        {
            "filename": [str(name) for name in filenames],
            "prob": [float(p) for p in probs],
            "label_true": [int(lbl) for lbl in labels_true],
        }
    )
    oof_df.to_csv(oof_path, index=False)
    logger.info("Saved OOF predictions => %s", oof_path)

    fold_idx = 0 if args.fold is None else int(args.fold)
    cv_path = Path("reports/cv_results.csv")
    cv_path.parent.mkdir(parents=True, exist_ok=True)

    if cv_path.exists():
        cv_df = pd.read_csv(cv_path)
        if "fold" in cv_df.columns:
            cv_df = cv_df[cv_df["fold"] != fold_idx]
        else:
            cv_df = pd.DataFrame(columns=["fold", "macro_f1"])
    else:
        cv_df = pd.DataFrame(columns=["fold", "macro_f1"])

    updated = pd.concat(
        [cv_df, pd.DataFrame([{"fold": fold_idx, "macro_f1": float(result.macro_f1)}])],
        ignore_index=True,
    )
    updated = updated.sort_values("fold").reset_index(drop=True)
    updated["mean_macro_f1"] = updated["macro_f1"].mean() if not updated.empty else float("nan")
    updated["std_macro_f1"] = (
        updated["macro_f1"].std(ddof=0) if len(updated) > 0 else float("nan")
    )
    updated.to_csv(cv_path, index=False)
    logger.info("Saved CV results => %s", cv_path)


if __name__ == "__main__":
    main()
