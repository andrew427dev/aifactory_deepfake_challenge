import argparse
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datamodules import create_dataloader
from src.evaluation.metrics import macro_f1
from src.models.loader import load_model_and_processor, resolve_model_source
from src.training.trainer import ExponentialMovingAverage, Trainer
from src.utils.device import resolve_device, should_enable_amp
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training entry point")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training config")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="Override runtime.device")
    parser.add_argument("--disable-amp", action="store_true", help="Force disable AMP regardless of config")
    return parser.parse_args()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    runtime_cfg = cfg.get("runtime", {})
    device_pref = args.device if args.device is not None else runtime_cfg.get("device", "auto")
    device = resolve_device(device_pref)
    logger.info("Using device: %s", device.type)

    amp_requested = runtime_cfg.get("amp", True)
    if args.disable_amp:
        amp_requested = False
    amp_enabled = should_enable_amp(device, amp_requested)
    logger.info("AMP enabled: %s", amp_enabled)

    model_source = resolve_model_source(model_cfg)
    logger.info("Loading model resources from %s", model_source)

    model, processor, is_fallback = load_model_and_processor(model_cfg)
    if is_fallback:
        logger.warning("Using fallback simple model; pretrained weights were unavailable.")

    trainer = Trainer(model=model, device=device, amp=amp_enabled, logger=logger)
    optimizer_cfg = cfg.get("optimizer", {})
    optimizer = trainer.configure_optimizer(optimizer_cfg)

    scheduler_cfg = cfg.get("scheduler", {})
    trainer_cfg = cfg.get("trainer", {})
    dataloader_cfg = cfg.get("dataloader", {})
    runtime_paths = cfg.get("paths", {})

    epochs = max(1, int(trainer_cfg.get("epochs", trainer_cfg.get("max_epochs", 1))))
    grad_accum = max(1, int(trainer_cfg.get("gradient_accumulation_steps", 1)))
    clip_grad = float(trainer_cfg.get("clip_grad_norm", 0.0) or 0.0)
    log_interval = int(trainer_cfg.get("log_interval", 50))

    num_labels = getattr(model.config, "num_labels", 2)
    image_size = getattr(model.config, "image_size", 224)

    train_manifest = runtime_paths.get("train_manifest")
    val_manifest = runtime_paths.get("val_manifest")
    fallback_train = int(trainer_cfg.get("fallback_train_size", 128))
    fallback_val = int(trainer_cfg.get("fallback_val_size", 64))

    train_loader = create_dataloader(
        train_manifest,
        processor,
        dataloader_cfg.get("train", {}),
        fallback_length=fallback_train,
        default_image_size=image_size,
        default_num_labels=num_labels,
        seed=seed,
    )

    val_loader = create_dataloader(
        val_manifest,
        processor,
        dataloader_cfg.get("val", {}),
        fallback_length=fallback_val,
        default_image_size=image_size,
        default_num_labels=num_labels,
        seed=seed,
    )

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = trainer.create_scheduler(scheduler_cfg, total_steps) if scheduler_cfg else None

    ema_cfg = cfg.get("ema", {})
    ema_enabled = bool(ema_cfg.get("enable", ema_cfg.get("enabled", False)))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema = ExponentialMovingAverage(trainer.model, decay=ema_decay) if ema_enabled else None
    trainer.set_ema(ema)

    criterion = trainer.criterion

    def evaluate(loader) -> Tuple[float, float]:
        if loader is None:
            return 0.0, 0.0

        context = ema.average_parameters(trainer.model) if ema is not None else nullcontext()
        was_training = trainer.model.training
        trainer.model.eval()
        losses: List[float] = []
        preds: List[int] = []
        labels_all: List[int] = []
        with torch.no_grad(), context:
            for batch in loader:
                batch = trainer.prepare_batch(batch)
                labels = batch.pop("labels")
                with torch.cuda.amp.autocast(enabled=trainer.amp):
                    outputs = trainer.model(**batch)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = criterion(logits, labels)
                losses.append(float(loss.item()))
                preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
                labels_all.extend(labels.detach().cpu().tolist())
        if was_training:
            trainer.model.train()
        val_loss = float(sum(losses) / max(1, len(losses)))
        val_f1 = float(macro_f1(labels_all, preds)) if labels_all else 0.0
        return val_loss, val_f1

    best_f1 = -float("inf")
    output_dir = Path(runtime_paths.get("output_dir", "experiments/exp001"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.ckpt"

    logger.info("Starting training for %d epochs (%d optimizer steps per epoch)", epochs, steps_per_epoch)

    for epoch in range(1, epochs + 1):
        trainer.model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = trainer.prepare_batch(batch)
            labels = batch.pop("labels")
            with torch.cuda.amp.autocast(enabled=trainer.amp):
                outputs = trainer.model(**batch)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = criterion(logits, labels)
            loss_value = float(loss.item())
            running_loss += loss_value
            loss = loss / grad_accum
            trainer.scaler.scale(loss).backward()

            should_step = (step % grad_accum == 0) or (step == len(train_loader))
            if should_step:
                trainer.scaler.unscale_(optimizer)
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), clip_grad)
                trainer.scaler.step(optimizer)
                trainer.scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(trainer.model)

            if log_interval > 0 and step % log_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                avg_loss = running_loss / step
                logger.info(
                    "Epoch %d Step %d/%d | loss=%.4f | lr=%.6f", epoch, step, len(train_loader), avg_loss, current_lr
                )

        train_loss = running_loss / max(1, len(train_loader))
        val_loss, val_f1 = evaluate(val_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d completed | train_loss=%.4f | val_loss=%.4f | val_macro_f1=%.4f | lr=%.6f",
            epoch,
            train_loss,
            val_loss,
            val_f1,
            current_lr,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            trainer.save_checkpoint(best_path)
            logger.info("New best Macro F1: %.4f (epoch %d) -> saved to %s", best_f1, epoch, best_path)

    logger.info("Training finished. Best Macro F1: %.4f", best_f1)
    logger.info("Best checkpoint saved at %s", best_path)


if __name__ == "__main__":
    main()
