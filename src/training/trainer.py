import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast


class Trainer:
    """Minimal trainer supporting AMP and device placement."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        amp: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.amp = bool(amp) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.amp)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self._ema: Optional["ExponentialMovingAverage"] = None

    def configure_optimizer(self, optimizer_cfg: Dict) -> torch.optim.Optimizer:
        name = optimizer_cfg.get("name", "adamw").lower()
        lr = optimizer_cfg.get("lr", 5.0e-5)
        weight_decay = optimizer_cfg.get("weight_decay", 0.0)
        betas = optimizer_cfg.get("betas")

        if name == "adamw":
            kwargs = {"lr": lr, "weight_decay": weight_decay}
            if betas is not None:
                kwargs["betas"] = tuple(betas)
            optimizer = torch.optim.AdamW(self.model.parameters(), **kwargs)
        elif name == "sgd":
            momentum = optimizer_cfg.get("momentum", 0.9)
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:  # pragma: no cover - defensive path
            raise ValueError(f"Unsupported optimizer: {name}")

        self.optimizer = optimizer
        return optimizer

    def set_ema(self, ema: Optional["ExponentialMovingAverage"]) -> None:
        self._ema = ema

    @property
    def ema(self) -> Optional["ExponentialMovingAverage"]:
        return self._ema

    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device, non_blocking=True)
            else:
                prepared[key] = value
        return prepared

    def _generate_dummy_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch_size = max(1, batch_size)
        num_labels = getattr(self.model.config, "num_labels", 2)
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=self.device)
        labels = torch.randint(0, num_labels, (batch_size,), device=self.device)
        return {"pixel_values": pixel_values, "labels": labels}

    def _forward_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        labels = batch["labels"]
        with autocast(enabled=self.amp):
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = self.criterion(logits, labels)
        return loss

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._forward_loss(batch)

    def predict_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        with autocast(enabled=self.amp):
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
        return logits

    def create_scheduler(self, scheduler_cfg: Dict, total_steps: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if not scheduler_cfg:
            return None

        name = scheduler_cfg.get("name", "").lower()
        if not name:
            return None

        if total_steps <= 0:
            raise ValueError("total_steps must be positive when creating a scheduler")

        if name == "cosine":
            warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
            min_lr = float(scheduler_cfg.get("min_lr", 0.0))
            base_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 1.0

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return (step + 1) / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                min_factor = min_lr / base_lr if base_lr > 0 else 0.0
                return max(min_factor, cosine)

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        if name == "onecycle":
            max_lr = float(scheduler_cfg.get("max_lr", self.optimizer.param_groups[0]["lr"]))
            pct_start = float(scheduler_cfg.get("pct_start", 0.3))
            anneal_strategy = scheduler_cfg.get("anneal_strategy", "cos")
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                anneal_strategy=anneal_strategy,
            )

        raise ValueError(f"Unsupported scheduler: {name}")

    def fit_dummy(self, epochs: int, steps_per_epoch: int, batch_size: int) -> None:
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been configured. Call configure_optimizer first.")

        self.model.train()
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                batch = self._generate_dummy_batch(batch_size)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._forward_loss(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.logger.info("Epoch %d Step %d - loss: %.4f", epoch + 1, step + 1, float(loss.item()))

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: Path, strict: bool = False) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state, strict=strict)

    def to(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)
        self.amp = self.amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.amp)


class ExponentialMovingAverage:
    """Simple EMA implementation tracking model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
        self.backup: Dict[str, torch.Tensor] | None = None

    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        self.apply(model)
        try:
            yield
        finally:
            self.restore(model)

    def apply(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        if self.backup is None:
            return
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = None
