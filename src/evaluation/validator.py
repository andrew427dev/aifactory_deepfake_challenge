from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

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

    def _positive_class_probability(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        if logits.ndim == 1:
            # Binary logits of shape (2,) are handled by softmax below
            logits = logits.unsqueeze(0)
        if logits.shape[-1] == 1:
            values = torch.sigmoid(logits.squeeze(-1))
            return values if values.ndim > 0 else values.unsqueeze(0)
        index = 1 if logits.shape[-1] > 1 else 0
        probs = F.softmax(logits, dim=-1)[..., index]
        return probs

    def _expand_filenames(self, value, expected: int, fallback_start: int) -> List[str]:
        if value is None:
            return [f"sample_{i:05d}" for i in range(fallback_start, fallback_start + expected)]
        if isinstance(value, (list, tuple)):
            items = [str(v) for v in value]
        elif isinstance(value, torch.Tensor):
            items = [str(v.item()) for v in value.view(-1)]
        else:
            items = [str(value)]
        if len(items) != expected:
            if len(items) == 1:
                items = items * expected
            else:
                self.logger.warning(
                    "Filename count mismatch (got %d expected %d); appending fallbacks.",
                    len(items),
                    expected,
                )
                items = items[:expected]
                if len(items) < expected:
                    remaining = expected - len(items)
                    items.extend(
                        f"sample_{i:05d}" for i in range(fallback_start, fallback_start + remaining)
                    )
        return items

    def _accumulate_predictions(
        self,
        dataloader: Iterable[Dict[str, torch.Tensor]],
    ) -> Tuple[ValidationResult, Dict[str, List]]:
        self.model.eval()
        losses = []
        preds: list[int] = []
        labels: list[int] = []
        probs: list[float] = []
        filenames: list[str] = []
        fallback_counter = 0

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
                detached = logits.detach()
                preds.extend(torch.argmax(detached, dim=-1).cpu().tolist())
                batch_labels = targets.detach().cpu().tolist()
                labels.extend(batch_labels)

                prob_tensor = self._positive_class_probability(detached).cpu().view(-1)
                probs.extend(float(v) for v in prob_tensor.tolist())

                batch_filenames = batch.get("filename")
                filenames.extend(
                    self._expand_filenames(batch_filenames, len(batch_labels), fallback_counter)
                )
                fallback_counter += len(batch_labels)

        avg_loss = float(sum(losses) / max(1, len(losses)))
        macro = float(macro_f1(labels, preds)) if labels else 0.0
        details = {
            "filenames": filenames,
            "probs": probs,
            "labels": labels,
            "preds": preds,
        }
        return ValidationResult(loss=avg_loss, macro_f1=macro), details

    def evaluate(
        self, dataloader: DataLoader, return_details: bool = False
    ) -> ValidationResult | Tuple[ValidationResult, Dict[str, List]]:
        result, details = self._accumulate_predictions(dataloader)
        self.logger.info("Validation | loss=%.4f | macro_f1=%.4f", result.loss, result.macro_f1)
        if return_details:
            return result, details
        return result
