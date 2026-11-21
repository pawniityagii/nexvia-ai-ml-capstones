"""Image loading + preprocessing shared by training and serving.

Produces the DenseNet input tensor (resize -> 3-channel -> ImageNet-normalise)
as a plain numpy array so this module has no torch dependency; the torch side
wraps it in a tensor. Handles PNG/JPG and DICOM inputs identically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import config as C
from . import dicom_io


def load_image(path: str | Path) -> np.ndarray:
    """Load any supported file to a float32 grayscale array in [0, 1]."""
    if dicom_io.is_dicom(path):
        return dicom_io.read_dicom(path)
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def to_model_input(arr: np.ndarray, size: int = C.IMG_SIZE) -> np.ndarray:
    """Grayscale [0,1] -> normalised 3xHxW float32 (ImageNet stats)."""
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
    img = img.resize((size, size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.stack([x, x, x], axis=0)                      # 1->3 channels
    mean = np.array(C.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(C.IMAGENET_STD, dtype=np.float32)[:, None, None]
    return (x - mean) / std


def augment(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light train-time augmentation (numpy): flip, small rotation, brightness."""
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    angle = float(rng.uniform(-8, 8))
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    out = np.asarray(img, dtype=np.float32) / 255.0
    out = np.clip(out * rng.uniform(0.9, 1.1), 0, 1)     # brightness jitter
    return out
