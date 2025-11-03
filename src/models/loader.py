# src/models/loader.py
from __future__ import annotations
import logging, os
from typing import Dict, Optional, Tuple
import torch
import numpy as np

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
        self.size = size
        self.mean = torch.tensor(mean or [0.485,0.456,0.406]).view(3,1,1)
        self.std  = torch.tensor(std  or [0.229,0.224,0.225]).view(3,1,1)
    def __call__(self, img):
        # img: PIL.Image (H,W,3)
        import torchvision.transforms as T
        tfm = T.Compose([
            T.Resize(self.size),
            T.CenterCrop(self.size),
            T.ToTensor(),
            T.Normalize(self.mean.flatten().tolist(), self.std.flatten().tolist()),
        ])
        out = tfm(img)
        return {"pixel_values": out.unsqueeze(0)}  # BCHW

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

def load_model_and_processor(model_cfg: Dict) -> Tuple[torch.nn.Module, object, bool]:
    """
    Returns: model, processor, is_fallback
    """
    _enable_perf_flags()
    model_source = resolve_model_source(model_cfg)
    name = model_cfg.get("model", {}).get("name", "")
    num_labels = int(model_cfg.get("model", {}).get("num_labels", 2))
    feat = model_cfg.get("feature_extractor", {})
    image_size = int(feat.get("size", 224))
    mean = feat.get("image_mean", [0.485, 0.456, 0.406])
    std  = feat.get("image_std",  [0.229, 0.224, 0.225])

    # 1) timm 경로 (예: timm/efficientnet_b4)
    if name.startswith("timm/"):
        if timm is None:
            raise ImportError("timm가 필요합니다. pip install timm")
        timm_name = name.split("timm/")[-1]
        model = timm.create_model(timm_name, pretrained=True, num_classes=num_labels)
        processor = SimpleImageProcessor(size=image_size, mean=mean, std=std)
        return model, processor, False

    # 2) transformers 경로 (예: google/vit-base-patch16-224)
    try:
        processor = AutoImageProcessor.from_pretrained(model_source)
        model = AutoModelForImageClassification.from_pretrained(model_source, num_labels=num_labels)
        return model, processor, False
    except Exception as exc:
        logger.warning("transformers 로드 실패(%s). fallback 간단분류기 사용.", exc)

    # 3) fallback
    model = SimpleClassifier(image_size=image_size, num_labels=num_labels)
    processor = SimpleImageProcessor(size=image_size, mean=mean, std=std)
    return model, processor, True
