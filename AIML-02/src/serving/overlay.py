"""Render a Grad-CAM/saliency heatmap as a colour overlay on the scan.

Pure numpy + PIL (no OpenCV dependency). Produces a base64 PNG the React viewer
can display directly, plus a standalone heatmap PNG.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def _jet(x: np.ndarray) -> np.ndarray:
    """Minimal jet-like colormap for a [0,1] array -> uint8 RGB."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_png(gray01: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> bytes:
    """Blend heatmap over the grayscale scan; return PNG bytes."""
    size = heatmap.shape[0]
    base = Image.fromarray((np.clip(gray01, 0, 1) * 255).astype(np.uint8), "L") \
        .resize((size, size), Image.BILINEAR).convert("RGB")
    base_arr = np.asarray(base, dtype=np.float32)
    heat = _jet(heatmap).astype(np.float32)
    mask = np.clip(heatmap, 0, 1)[..., None]            # fade where heat is low
    blended = base_arr * (1 - alpha * mask) + heat * (alpha * mask)
    out = Image.fromarray(blended.astype(np.uint8), "RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def to_base64_png(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()
