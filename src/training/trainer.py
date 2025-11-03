from __future__ import annotations

import logging
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

    def configure_optimizer(self, optimizer_cfg: Dict) -> torch.optim.Optimizer:
        name = optimizer_cfg.get("name", "adamw").lower()
        lr = optimizer_cfg.get("lr", 5.0e-5)
        weight_decay = optimizer_cfg.get("weight_decay", 0.0)

        if name == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            momentum = optimizer_cfg.get("momentum", 0.9)
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:  # pragma: no cover - defensive path
            raise ValueError(f"Unsupported optimizer: {name}")

        self.optimizer = optimizer
        return optimizer

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
