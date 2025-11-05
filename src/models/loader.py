# src/models/loader.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# transformers
from transformers import AutoImageProcessor, AutoModelForImageClassification

# timm
try:
    import timm
except Exception:
    timm = None

class SimpleBatch(dict):
    def to(self, device: torch.device) -> "SimpleBatch":
        b = SimpleBatch({k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.items()})
        return b

class SimpleImageProcessor:
    def __init__(self, size=224, mean=None, std=None):
        from torchvision import transforms as T  # local import keeps dependency optional until needed

        if isinstance(size, (tuple, list)):
            if len(size) != 2:
                raise ValueError("size must be an int or a length-2 sequence")
            self.size = (int(size[0]), int(size[1]))
        else:
            self.size = int(size)

        default_mean = [0.485, 0.456, 0.406]
        default_std = [0.229, 0.224, 0.225]
        self.mean = [float(m) for m in (mean or default_mean)]
        self.std = [float(s) for s in (std or default_std)]
        self._transform = T.Compose(
            [
                T.Resize(self.size),
                T.CenterCrop(self.size),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

    @staticmethod
    def _ensure_pil(image):
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("numpy arrays must have shape HxWx3")
            array = image
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            return Image.fromarray(array)
        raise TypeError("Unsupported image type for SimpleImageProcessor")

    def __call__(self, images, return_tensors: Optional[str] = "pt", **_kwargs):
        if return_tensors not in (None, "pt"):
            raise ValueError("SimpleImageProcessor only supports return_tensors='pt' or None")

        batch = list(images) if isinstance(images, (list, tuple)) else [images]
        tensors = [self._transform(self._ensure_pil(img)) for img in batch]
        stacked = torch.stack(tensors, dim=0)
        return {"pixel_values": stacked}

class SimpleClassifier(torch.nn.Module):
    def __init__(self, image_size=224, num_labels=2):
        super().__init__()
        c = 3
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(c, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.head = torch.nn.Linear(64, num_labels)
    def forward(self, x):
        z = self.net(x).flatten(1)
        return self.head(z)

def _enable_perf_flags():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

def resolve_model_source(model_cfg: Dict) -> str:
    # 우선순위: model.checkpoint -> model.name
    ckpt = model_cfg.get("checkpoint")
    if ckpt and os.path.exists(ckpt):
        return ckpt
    return model_cfg.get("model", {}).get("name", "google/vit-base-patch16-224")


def _infer_feature_defaults(identifier: str) -> Dict[str, object]:
    lower = (identifier or "").lower()
    defaults = {"size": 224, "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]}
    if "efficientnet_b4" in lower:
        defaults["size"] = 380
    elif "vit" in lower or "swin" in lower:
        defaults["size"] = 224
    return defaults


def _hf_cache_dir_hint() -> str:
    for env_var in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.getenv(env_var)
        if value:
            return value
    return str(Path.home() / ".cache" / "huggingface")


def _load_transformers_with_retry(model_source: str, num_labels: int, attempts: int = 2):
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            processor = AutoImageProcessor.from_pretrained(model_source)
            model = AutoModelForImageClassification.from_pretrained(model_source, num_labels=num_labels)
            return processor, model
        except Exception as exc:  # pragma: no cover - depends on network availability
            last_exc = exc
            if attempt < attempts:
                logger.warning(
                    "transformers 자원 로드 실패 시도 %d/%d (%s). 재시도합니다.",
                    attempt,
                    attempts,
                    exc,
                )
            else:
                cache_hint = _hf_cache_dir_hint()
                logger.error(
                    "transformers 사전학습 가중치 로드 실패(%s). 필요한 경우 %s 경로에 수동으로 다운로드하세요.",
                    exc,
                    cache_hint,
                )
    if last_exc is None:
        raise RuntimeError("Unknown error while loading transformers resources")
    raise last_exc

def load_model_and_processor(model_cfg: Dict) -> Tuple[torch.nn.Module, object, bool]:
    """
    Returns: model, processor, is_fallback
    """
    _enable_perf_flags()
    model_source = resolve_model_source(model_cfg)
    name = model_cfg.get("model", {}).get("name", "")
    num_labels = int(model_cfg.get("model", {}).get("num_labels", 2))
    feat = model_cfg.get("feature_extractor", {})
    identifier = name or model_source
    defaults = _infer_feature_defaults(identifier)

    image_size = int(feat.get("size", defaults["size"]))
    mean = list(feat.get("image_mean", defaults["image_mean"]))
    std = list(feat.get("image_std", defaults["image_std"]))

    # 1) timm 경로 (예: timm/efficientnet_b4)
    if name.startswith("timm/"):
        if timm is None:
            raise ImportError("timm가 필요합니다. pip install timm")
        timm_name = name.split("timm/")[-1]
        try:
            model = timm.create_model(timm_name, pretrained=True, num_classes=num_labels)
            processor = SimpleImageProcessor(size=image_size, mean=mean, std=std)
            return model, processor, False
        except Exception as exc:  # pragma: no cover - network/env dependent
            logger.warning("timm 모델 로드 실패(%s). 간단 분류기로 대체합니다.", exc)

    # 2) transformers 경로 (예: google/vit-base-patch16-224)
    try:
        processor, model = _load_transformers_with_retry(model_source, num_labels)
        return model, processor, False
    except Exception as exc:
        logger.warning("transformers 로드 실패(%s). fallback 간단분류기 사용.", exc)

    # 3) fallback
    model = SimpleClassifier(image_size=image_size, num_labels=num_labels)
    processor = SimpleImageProcessor(size=image_size, mean=mean, std=std)
    return model, processor, True
