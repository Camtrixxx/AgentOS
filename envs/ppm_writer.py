from __future__ import annotations

from pathlib import Path


def write_ppm(path: Path, image) -> None:
    """Write a numpy RGB image array (H, W, 3) as a binary P6 PPM file."""
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {image.shape}")
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + image.tobytes())
