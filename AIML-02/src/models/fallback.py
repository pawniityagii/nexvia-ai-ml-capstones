"""Deterministic fallback predictor - NO torch, NO trained weights.

This exists so the full application (upload -> inference -> heatmap -> review ->
audit) runs with zero download and zero GPU, for the demo and CI. It is a
transparent image-processing heuristic, NOT a diagnostic model and NOT a neural
network: it looks for bright opacities inside an approximate lung mask and
reports how strong they are. The real DenseNet replaces it once trained
(see predictor.py). Responses from this path are tagged mode="demo-heuristic".
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.filters import gaussian

from ..common import config as C


def _lung_mask(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size] / size
    mask = np.zeros((size, size), dtype=bool)
    for cx in (0.33, 0.67):
        mask |= (((xx - cx) ** 2) / 0.02 + ((yy - 0.5) ** 2) / 0.06) < 1.0
    return mask


def predict(gray01: np.ndarray) -> dict:
    """gray01: HxW float in [0,1]. Returns prob/label/heatmap(HxW in [0,1])."""
    img = np.asarray(Image.fromarray((np.clip(gray01, 0, 1) * 255).astype(np.uint8))
                     .resize((C.IMG_SIZE, C.IMG_SIZE), Image.BILINEAR),
                     dtype=np.float32) / 255.0
    background = gaussian(img, sigma=30, preserve_range=True)
    excess = np.clip(img - background - 0.02, 0, None)      # local bright opacity
    mask = _lung_mask(C.IMG_SIZE)
    sal = excess * mask
    if sal.max() > 0:
        sal = sal / sal.max()
    # probability: peak opacity strength in the lung fields, calibrated on the
    # synthetic normal/abnormal score distributions (see fallback tuning).
    masked = excess[mask]
    score = float(np.percentile(masked, 99.5)) if masked.size else 0.0
    prob = float(1 / (1 + np.exp(-(score - 0.09) / 0.025)))
    label = C.CLASSES[1] if prob >= C.DEFAULT_THRESHOLD else C.CLASSES[0]
    return {"probability": prob, "label": label, "heatmap": sal,
            "mode": "demo-heuristic"}
