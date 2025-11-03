# src/preprocessing/face.py
from __future__ import annotations
import os
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch

try:
    from facenet_pytorch import MTCNN
except Exception as e:
    MTCNN = None

class FaceAligner:
    """
    간단한 얼굴 검출/정렬 래퍼.
    - detector='mtcnn'만 지원(가볍고 설치 쉬움)
    - 입력: PIL.Image
    - 출력: 얼굴만 crop된 PIL.Image (없으면 원본 리사이즈)
    """
    def __init__(self, detector: str = "mtcnn", device: Optional[str] = None, image_size: int = 224):
        self.image_size = int(image_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        detector = (detector or "mtcnn").lower()
        if detector != "mtcnn":
            raise ValueError("Only 'mtcnn' is supported in this baseline.")
        if MTCNN is None:
            raise ImportError("facenet-pytorch가 필요합니다. pip install facenet-pytorch")
        self.mtcnn = MTCNN(image_size=self.image_size, margin=20, post_process=True, device=self.device)

    def __call__(self, img: Image.Image) -> Image.Image:
        # 얼굴이 없으면 None 반환 → 호출측에서 원본 리사이즈로 fallback
        with torch.inference_mode():
            face = self.mtcnn(img, save_path=None, return_prob=False)
        if face is None:
            return img.resize((self.image_size, self.image_size))
        # face: torch.Tensor [3, H, W], 0~1 float
        face = (face.clamp(0, 1) * 255).byte().cpu().numpy()
        face = np.transpose(face, (1, 2, 0))  # HWC
        return Image.fromarray(face)
