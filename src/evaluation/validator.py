from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from .metrics import macro_f1


@dataclass(frozen=True)
class ValidationResult:
    """Container storing validation loss and macro F1."""

    loss: float
    macro_f1: float


class Validator:
    """Run evaluation on a dataloader and compute Macro F1."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        amp: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(self.device)
        self.amp = bool(amp) and self.device.type == "cuda"
        self.criterion = torch.nn.CrossEntropyLoss()

    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prepared: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device, non_blocking=True)
            else:
                prepared[key] = value
        return prepared

    def _accumulate_predictions(
        self,
        dataloader: Iterable[Dict[str, torch.Tensor]],
    ) -> ValidationResult:
        self.model.eval()
        losses = []
        preds: list[int] = []
        labels: list[int] = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self.prepare_batch(batch)
                inputs = batch.get("x")
                targets = batch.get("y")
                if inputs is None or targets is None:
                    raise KeyError("Batch must contain 'x' and 'y' tensors for validation.")

                with torch.cuda.amp.autocast(enabled=self.amp):
                    outputs = self.model(pixel_values=inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = self.criterion(logits, targets)

                losses.append(float(loss.item()))
                preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
                labels.extend(targets.detach().cpu().tolist())

        avg_loss = float(sum(losses) / max(1, len(losses)))
        macro = float(macro_f1(labels, preds)) if labels else 0.0
        return ValidationResult(loss=avg_loss, macro_f1=macro)

    def evaluate(self, dataloader: DataLoader) -> ValidationResult:
        result = self._accumulate_predictions(dataloader)
        self.logger.info("Validation | loss=%.4f | macro_f1=%.4f", result.loss, result.macro_f1)
        return result
