"""Image dataset utilities for deepfake classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ID_TO_LABEL = {0: "Real", 1: "Fake"}
LABEL_TO_ID = {label.lower(): idx for idx, label in ID_TO_LABEL.items()}

_PATH_CANDIDATES: Tuple[str, ...] = (
    "filepath",
    "file_path",
    "path",
    "image_path",
    "frame_path",
    "filename",
)
_LABEL_CANDIDATES: Tuple[str, ...] = ("label", "target", "y", "class", "classname")


@dataclass(frozen=True)
class ImagePreprocessConfig:
    image_size: int = 224
    mean: Sequence[float] = (0.5, 0.5, 0.5)
    std: Sequence[float] = (0.5, 0.5, 0.5)


def _detect_column(columns: Iterable[str], candidates: Tuple[str, ...]) -> Optional[str]:
    for name in candidates:
        if name in columns:
            return name
    return None


def _as_tensor_mean_std(values: Sequence[float]) -> torch.Tensor:
    array = torch.tensor(list(values), dtype=torch.float32).view(-1, 1, 1)
    if array.numel() != 3:
        raise ValueError("Mean/Std must contain three values for RGB images")
    return array


def _normalize_image(image: Image.Image, config: ImagePreprocessConfig) -> torch.Tensor:
    resized = image.resize((config.image_size, config.image_size), resample=Image.BILINEAR)
    tensor = torch.from_numpy(np.array(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0
    mean = _as_tensor_mean_std(config.mean)
    std = _as_tensor_mean_std(config.std)
    return (tensor - mean) / std


class ImageDataset(Dataset):
    """Dataset reading RGB images and integer labels from a CSV manifest."""

    def __init__(
        self,
        manifest_path: Path | str,
        preprocess: Optional[ImagePreprocessConfig] = None,
        path_column: Optional[str] = None,
        label_column: Optional[str] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.frame = pd.read_csv(self.manifest_path)
        if self.frame.empty:
            raise ValueError(f"Manifest {self.manifest_path} is empty")

        self.preprocess = preprocess or ImagePreprocessConfig()
        columns = list(self.frame.columns)
        self.path_column = path_column or _detect_column(columns, _PATH_CANDIDATES)
        if self.path_column is None:
            raise KeyError(
                f"Manifest must contain one of path columns {list(_PATH_CANDIDATES)}; found {columns}"
            )
        self.label_column = label_column or _detect_column(columns, _LABEL_CANDIDATES)
        if self.label_column is None:
            raise KeyError(
                f"Manifest must contain one of label columns {list(_LABEL_CANDIDATES)}; found {columns}"
            )

        self.root_dir = self.manifest_path.parent
        self.samples: List[Tuple[Path, int, str]] = []
        for _, row in self.frame.iterrows():
            raw_path = str(row[self.path_column])
            rel_path = Path(raw_path)
            full_path = rel_path if rel_path.is_absolute() else self.root_dir / rel_path
            label_value = self._encode_label(row[self.label_column])
            identifier = str(rel_path.name if rel_path.name else rel_path)
            self.samples.append((full_path, label_value, identifier))

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_label(self, value) -> int:
        if isinstance(value, (int, np.integer)):
            int_value = int(value)
            if int_value not in ID_TO_LABEL:
                raise ValueError(f"Unsupported label id {int_value}; expected 0 for Real or 1 for Fake")
            return int_value
        label_str = str(value).strip().lower()
        if label_str not in LABEL_TO_ID:
            raise ValueError(f"Unsupported label name '{value}'; expected one of {list(ID_TO_LABEL.values())}")
        return LABEL_TO_ID[label_str]

    def _load_image(self, path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as img:
            return img.convert("RGB")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path, label, identifier = self.samples[index]
        image_rgb = self._load_image(path)
        tensor = _normalize_image(image_rgb, self.preprocess)
        return {
            "x": tensor,
            "y": torch.tensor(label, dtype=torch.long),
            "filename": identifier,
        }


def create_image_dataloader(
    manifest_path: Path | str,
    loader_cfg: Optional[Dict] = None,
    preprocess: Optional[ImagePreprocessConfig] = None,
    path_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> DataLoader:
    loader_cfg = loader_cfg or {}
    batch_size = int(loader_cfg.get("batch_size", 32))
    num_workers = int(loader_cfg.get("num_workers", 4))
    shuffle = bool(loader_cfg.get("shuffle", False))
    pin_memory = bool(loader_cfg.get("pin_memory", False))

    dataset = ImageDataset(
        manifest_path=manifest_path,
        preprocess=preprocess,
        path_column=path_column,
        label_column=label_column,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
