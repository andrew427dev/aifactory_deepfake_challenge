"""Video dataset utilities for deepfake classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .dataset_image import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    ImagePreprocessConfig,
    _as_tensor_mean_std,
    _detect_column,
)

try:  # pragma: no cover - optional dependency for real video decoding
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully
    cv2 = None  # type: ignore

_VIDEO_PATH_CANDIDATES: Tuple[str, ...] = (
    "video_path",
    "filepath",
    "file_path",
    "path",
    "filename",
)


@dataclass(frozen=True)
class VideoPreprocessConfig(ImagePreprocessConfig):
    max_frames: int = 16


class VideoDataset(Dataset):
    """Dataset that loads RGB frames from videos with uniform sampling."""

    def __init__(
        self,
        manifest_path: Path | str,
        preprocess: Optional[VideoPreprocessConfig] = None,
        path_column: Optional[str] = None,
        label_column: Optional[str] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.frame = pd.read_csv(self.manifest_path)
        if self.frame.empty:
            raise ValueError(f"Manifest {self.manifest_path} is empty")

        self.preprocess = preprocess or VideoPreprocessConfig()
        columns = list(self.frame.columns)
        self.path_column = path_column or _detect_column(columns, _VIDEO_PATH_CANDIDATES)
        if self.path_column is None:
            raise KeyError(
                f"Manifest must contain one of path columns {list(_VIDEO_PATH_CANDIDATES)}; found {columns}"
            )
        self.label_column = label_column or _detect_column(columns, ("label", "target", "y", "class", "classname"))
        if self.label_column is None:
            raise KeyError(
                "Manifest must contain a label column such as 'label', 'target', 'y', 'class', or 'classname'"
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

        if self.preprocess.max_frames <= 0:
            raise ValueError("max_frames must be positive")

        self.mean = _as_tensor_mean_std(self.preprocess.mean)
        self.std = _as_tensor_mean_std(self.preprocess.std)

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

    def _read_video(self, path: Path) -> List[np.ndarray]:
        if path.is_dir():
            frames: List[np.ndarray] = []
            for file in sorted(path.iterdir()):
                if file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                with Image.open(file) as img:
                    frames.append(np.array(img.convert("RGB")))
            if not frames:
                raise ValueError(f"No image frames found in directory {path}")
            return frames

        if cv2 is None:
            raise ImportError(
                "opencv-python is required for reading video files; install opencv-python-headless in headless environments"
            )

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Video not found or unreadable: {path}")

        frames: List[np.ndarray] = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame[:, :, ::-1])  # BGR -> RGB
        capture.release()

        if not frames:
            raise ValueError(f"Video at {path} contains no frames")
        return frames

    def _sample_indices(self, length: int) -> np.ndarray:
        if length == 1:
            return np.zeros((self.preprocess.max_frames,), dtype=int)
        return np.linspace(0, length - 1, num=self.preprocess.max_frames, dtype=int)

    def _normalize_frame(self, frame: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(frame)
        resized = image.resize((self.preprocess.image_size, self.preprocess.image_size), resample=Image.BILINEAR)
        tensor = torch.from_numpy(np.array(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0
        return (tensor - self.mean) / self.std

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path, label, identifier = self.samples[index]
        frames = self._read_video(path)
        indices = self._sample_indices(len(frames))
        selected = [frames[int(i)] for i in indices]
        stacked = torch.stack([self._normalize_frame(frame) for frame in selected], dim=0)
        pooled = stacked.mean(dim=0)
        return {
            "x": pooled,
            "y": torch.tensor(label, dtype=torch.long),
            "filename": identifier,
        }


def create_video_dataloader(
    manifest_path: Path | str,
    loader_cfg: Optional[Dict] = None,
    preprocess: Optional[VideoPreprocessConfig] = None,
    path_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> DataLoader:
    loader_cfg = loader_cfg or {}
    batch_size = int(loader_cfg.get("batch_size", 8))
    num_workers = int(loader_cfg.get("num_workers", 4))
    shuffle = bool(loader_cfg.get("shuffle", False))
    pin_memory = bool(loader_cfg.get("pin_memory", False))

    dataset = VideoDataset(
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
