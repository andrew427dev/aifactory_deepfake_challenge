import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datamodules.dataset_image import ImageDataset, ImagePreprocessConfig
from src.datamodules.dataset_video import VideoDataset, VideoPreprocessConfig


def _write_manifest(path: Path, rows):
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_image_dataset_returns_normalized_tensor(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    rgb = np.full((32, 32, 3), 200, dtype=np.uint8)
    Image.fromarray(rgb).save(image_path)

    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"filepath": f"images/{image_path.name}", "label": "Real"},
        ],
    )

    dataset = ImageDataset(manifest, preprocess=ImagePreprocessConfig(image_size=16))
    sample = dataset[0]
    assert set(sample.keys()) == {"x", "y"}
    assert sample["x"].shape == (3, 16, 16)
    assert sample["x"].dtype == torch.float32
    assert sample["y"].item() == 0


def test_video_dataset_uniform_sampling(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(5):
        frame = np.full((32, 32, 3), i * 40, dtype=np.uint8)
        Image.fromarray(frame).save(frames_dir / f"frame_{i:02d}.png")

    manifest = tmp_path / "video_manifest.csv"
    _write_manifest(
        manifest,
        [
            {"filepath": frames_dir.name, "label": "Fake"},
        ],
    )

    dataset = VideoDataset(manifest, preprocess=VideoPreprocessConfig(image_size=16, max_frames=4))
    sample = dataset[0]
    assert sample["x"].shape == (3, 16, 16)
    assert sample["x"].dtype == torch.float32
    assert sample["y"].item() == 1
