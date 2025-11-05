"""Dataset utilities for manifest-driven training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _detect_column(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        if cand in columns:
            return cand
    return None


class ManifestImageDataset(Dataset):
    """Simple dataset that reads image paths and labels from a CSV manifest."""

    def __init__(
        self,
        manifest_path: Path,
        processor: Optional[object] = None,
        label_column: Optional[str] = None,
        path_column: Optional[str] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.processor = processor

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.frame = pd.read_csv(self.manifest_path)
        if self.frame.empty:
            raise ValueError(f"Manifest {self.manifest_path} is empty")

        cols = list(self.frame.columns)
        self.path_column = path_column or _detect_column(
            cols, ["filepath", "file_path", "path", "image_path", "frame_path", "filename"]
        )
        if self.path_column is None:
            raise KeyError(
                f"Manifest must contain one of path columns ['filepath', 'file_path', 'path', 'image_path', 'frame_path', 'filename']; found {cols}"
            )

        self.label_column = label_column or _detect_column(cols, ["label", "target", "y", "class"])
        if self.label_column is None:
            raise KeyError(
                f"Manifest must contain one of label columns ['label', 'target', 'y', 'class']; found {cols}"
            )

        self.root_dir = self.manifest_path.parent
        self.num_labels = int(self.frame[self.label_column].nunique())

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return len(self.frame)

    def _load_image(self, rel_path: str) -> np.ndarray:
        abs_path = Path(rel_path)
        if not abs_path.is_absolute():
            abs_path = self.root_dir / abs_path
        if not abs_path.exists():
            raise FileNotFoundError(f"Image not found: {abs_path}")
        with Image.open(abs_path) as img:
            return np.array(img.convert("RGB"))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        img_rgb = self._load_image(str(row[self.path_column]))
        label = int(row[self.label_column])

        if self.processor is not None:
            batch = self.processor(images=img_rgb, return_tensors="pt")
            if hasattr(batch, "items"):
                items = dict(batch.items())
            else:  # pragma: no cover - defensive branch
                items = dict(batch)
        else:
            tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            items = {"pixel_values": tensor.unsqueeze(0)}

        result = {}
        for key, value in items.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.squeeze(0)
        result["labels"] = torch.tensor(label, dtype=torch.long)
        return result


class RandomImageDataset(Dataset):
    """Fallback dataset that emits random tensors when manifests are missing."""

    def __init__(
        self,
        length: int,
        image_size: int = 224,
        num_labels: int = 2,
        processor: Optional[object] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.length = int(length)
        self.image_size = int(image_size)
        self.num_labels = int(num_labels)
        self.processor = processor
        self.rng = np.random.default_rng(seed) if seed is not None else None
        self.torch_generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.processor is not None:
            if self.rng is not None:
                random_image = self.rng.integers(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                random_image = np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
            batch = self.processor(images=random_image, return_tensors="pt")
            if hasattr(batch, "items"):
                items = dict(batch.items())
            else:  # pragma: no cover
                items = dict(batch)
            tensor = next(iter(items.values()))
            pixel_values = tensor.squeeze(0)
        else:
            if self.torch_generator is not None:
                pixel_values = torch.rand(3, self.image_size, self.image_size, generator=self.torch_generator)
            else:
                pixel_values = torch.rand(3, self.image_size, self.image_size)
        pixel_values = (pixel_values - 0.5) / 0.5
        if self.torch_generator is not None:
            label_tensor = torch.randint(
                0,
                self.num_labels,
                (1,),
                dtype=torch.long,
                generator=self.torch_generator,
            )
        else:
            label_tensor = torch.randint(0, self.num_labels, (1,), dtype=torch.long)
        label = label_tensor.item()
        return {
            "x": pixel_values,
            "y": torch.tensor(label, dtype=torch.long),
            "filename": f"random_{index:05d}",
        }


def create_dataloader(
    manifest_path: Optional[str],
    processor: Optional[object],
    loader_cfg: Dict,
    fallback_length: int,
    default_image_size: int,
    default_num_labels: int,
    seed: Optional[int] = None,
) -> DataLoader:
    batch_size = int(loader_cfg.get("batch_size", 8))
    num_workers = int(loader_cfg.get("num_workers", 4))
    shuffle = bool(loader_cfg.get("shuffle", False))
    pin_memory = bool(loader_cfg.get("pin_memory", False))

    if manifest_path and Path(manifest_path).exists():
        dataset = ManifestImageDataset(Path(manifest_path), processor=processor)
        logger.info("Loaded %d samples from %s", len(dataset), manifest_path)
    else:
        if manifest_path:
            logger.warning("Manifest not found at %s; falling back to random data for %d samples.", manifest_path, fallback_length)
        dataset = RandomImageDataset(
            length=max(1, fallback_length),
            image_size=default_image_size,
            num_labels=default_num_labels,
            processor=processor,
            seed=seed,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
