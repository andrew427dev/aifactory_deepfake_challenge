from typing import Optional

import torch

from src.utils.logging import get_logger

logger = get_logger("device")


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


_CHANNELS_LAST_ENABLED: bool = False


def enable_perf_flags(channels_last: bool, cudnn_benchmark: bool) -> None:
    """Configure global performance flags for the current process."""

    global _CHANNELS_LAST_ENABLED

    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    logger.info("cuDNN benchmark: %s", torch.backends.cudnn.benchmark)

    _CHANNELS_LAST_ENABLED = bool(channels_last)
    if _CHANNELS_LAST_ENABLED:
        logger.info("Tensor memory format: channels_last (apply when moving tensors to device)")
    else:
        logger.info("Tensor memory format: contiguous")


def channels_last_enabled() -> bool:
    """Return whether channels-last tensors were requested via enable_perf_flags."""

    return _CHANNELS_LAST_ENABLED
