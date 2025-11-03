import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def resolve_device(device_preference: Optional[str] = None) -> torch.device:
    """Return a torch.device based on preference and availability."""
    normalized = (device_preference or "auto").lower()
    if normalized in {"auto", "default"}:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    elif normalized in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        device = torch.device("cuda")
    elif normalized == "cpu":
        device = torch.device("cpu")
    else:
        try:
            device = torch.device(normalized)
        except Exception as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Unsupported device string: {device_preference}") from exc
    return device


def should_enable_amp(device: torch.device, requested: bool) -> bool:
    """Determine if AMP should be enabled for the given device."""
    return bool(requested) and device.type == "cuda"
