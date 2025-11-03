from __future__ import annotations

import logging
import os
import types
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger(__name__)


class SimpleBatch(dict):
    def to(self, device: torch.device) -> "SimpleBatch":
        return SimpleBatch({k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()})


class SimpleImageProcessor:
    def __init__(
        self,
        size: int = 224,
        mean: Optional[list[float]] = None,
        std: Optional[list[float]] = None,
    ) -> None:
        self.size = size
        self.mean = torch.tensor(mean or [0.5, 0.5, 0.5]).view(3, 1, 1)
        self.std = torch.tensor(std or [0.5, 0.5, 0.5]).view(3, 1, 1)

    def __call__(self, images, return_tensors: str = "pt") -> SimpleBatch:
        if return_tensors != "pt":  # pragma: no cover - defensive path
            raise ValueError("SimpleImageProcessor only supports return_tensors='pt'")
        if isinstance(images, np.ndarray):
            array = images
        else:
            array = np.array(images)
        resized = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        resized = torch.nn.functional.interpolate(
            resized.unsqueeze(0), size=(self.size, self.size), mode="bilinear", align_corners=False
        )[0]
        normalized = (resized - self.mean) / self.std
        return SimpleBatch({"pixel_values": normalized.unsqueeze(0)})


class SimpleClassifier(torch.nn.Module):
    def __init__(self, image_size: int = 224, num_labels: int = 2) -> None:
        super().__init__()
        hidden = 128
        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * image_size * image_size, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, num_labels),
        )
        self.config = types.SimpleNamespace(
            num_labels=num_labels,
            id2label={i: str(i) for i in range(num_labels)},
            label2id={str(i): i for i in range(num_labels)},
        )

    def forward(self, pixel_values: torch.Tensor):
        logits = self.network(pixel_values)
        return types.SimpleNamespace(logits=logits)


def resolve_model_source(model_cfg: Dict) -> str:
    checkpoint_path = model_cfg.get("checkpoint")
    if checkpoint_path:
        checkpoint_dir = os.path.dirname(checkpoint_path) or checkpoint_path
        if os.path.exists(checkpoint_dir):
            return checkpoint_dir
    model_section = model_cfg.get("model", {})
    return model_section.get("name", checkpoint_path or "google/vit-base-patch16-224")


def load_model_and_processor(model_cfg: Dict) -> Tuple[torch.nn.Module, Optional[object], bool]:
    model_source = resolve_model_source(model_cfg)
    allow_remote = model_cfg.get("allow_remote_download", False)
    try:
        processor = AutoImageProcessor.from_pretrained(model_source, local_files_only=not allow_remote)
        model = AutoModelForImageClassification.from_pretrained(
            model_source,
            num_labels=model_cfg.get("model", {}).get("num_labels"),
            id2label=model_cfg.get("model", {}).get("id2label"),
            label2id=model_cfg.get("model", {}).get("label2id"),
            local_files_only=not allow_remote,
        )
        return model, processor, False
    except Exception as exc:  # pragma: no cover - best effort fallback
        logger.warning(
            "Falling back to simple classifier because pretrained resources could not be loaded from %s: %s",
            model_source,
            exc,
        )
        image_size = model_cfg.get("feature_extractor", {}).get("size", 224)
        mean = model_cfg.get("feature_extractor", {}).get("image_mean", [0.5, 0.5, 0.5])
        std = model_cfg.get("feature_extractor", {}).get("image_std", [0.5, 0.5, 0.5])
        num_labels = model_cfg.get("model", {}).get("num_labels", 2)
        model = SimpleClassifier(image_size=image_size, num_labels=num_labels)
        processor = SimpleImageProcessor(size=image_size, mean=mean, std=std)
        return model, processor, True
