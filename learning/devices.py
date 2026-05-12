from __future__ import annotations

import torch


def resolve_torch_device(device: str) -> torch.device:
    """Resolve cpu/cuda/npu/auto without requiring torch_npu on CPU-only hosts."""

    requested = device.strip().lower()
    if requested == "auto":
        if _npu_available():
            return torch.device("npu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "npu":
        if not _npu_available():
            raise RuntimeError("NPU requested but torch_npu/torch.npu is not available")
        return torch.device("npu")
    return torch.device(requested)


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    npu = getattr(torch, "npu", None)
    is_available = getattr(npu, "is_available", None)
    return bool(callable(is_available) and is_available())

