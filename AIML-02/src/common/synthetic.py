"""Synthetic chest-X-ray generator.

Purpose: let the whole pipeline - preprocessing, training, Grad-CAM, serving,
viewer - run with ZERO download. These are procedurally drawn grayscale images
that merely *resemble* a frontal chest film; they are NOT real radiographs and
carry no clinical meaning. Real evaluation uses a public dataset (see
data/README.md); synthetic numbers are never reported as results.

Signal design: "abnormal" images get a bright Gaussian opacity blob in one lung
field (loosely mimicking a consolidation). The classifier learns normal vs
abnormal; Grad-CAM should localise the blob - a built-in sanity check for the
explanation heatmaps.
"""
from __future__ import annotations

import csv

import numpy as np
from PIL import Image

from . import config as C


def _chest_base(size: int, rng: np.random.Generator) -> np.ndarray:
    """A plausible frontal-chest intensity field in [0, 1]."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = np.full((size, size), 0.12, dtype=np.float32)

    # bright central mediastinum / spine column
    img += 0.45 * np.exp(-((xx - 0.5) ** 2) / (2 * 0.02))
    # two darker lung fields (air = dark on X-ray)
    for cx in (0.33, 0.67):
        img -= 0.18 * np.exp(-(((xx - cx) ** 2) / (2 * 0.02) +
                               ((yy - 0.5) ** 2) / (2 * 0.06)))
    # rib texture (faint horizontal bands)
    img += 0.05 * np.sin(yy * np.pi * 14) * (np.abs(xx - 0.5) > 0.12)
    # diaphragm / abdomen brighter at the bottom
    img += 0.30 * np.clip((yy - 0.72) / 0.28, 0, 1)
    # shoulders
    img += 0.20 * np.exp(-((yy - 0.12) ** 2) / (2 * 0.01))
    img += rng.normal(0, 0.02, (size, size))
    return np.clip(img, 0, 1)


def _add_opacity(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, tuple]:
    """Add one bright opacity blob in a lung field; return (img, bbox)."""
    size = img.shape[0]
    cx = rng.choice([0.33, 0.67]) + rng.normal(0, 0.03)
    cy = 0.5 + rng.normal(0, 0.08)
    r = rng.uniform(0.05, 0.10)
    intensity = rng.uniform(0.25, 0.45)
    yy, xx = np.mgrid[0:size, 0:size] / size
    blob = intensity * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * r ** 2))
    out = np.clip(img + blob, 0, 1)
    bbox = (int((cx - r) * size), int((cy - r) * size),
            int((cx + r) * size), int((cy + r) * size))
    return out, bbox


def generate_image(abnormal: bool, size: int = 512, seed: int | None = None):
    rng = np.random.default_rng(seed)
    img = _chest_base(size, rng)
    bbox = None
    if abnormal:
        img, bbox = _add_opacity(img, rng)
    arr = (img * 255).astype(np.uint8)
    return arr, bbox


def generate_dataset(n: int = 400, abnormal_frac: float = 0.5,
                     size: int = 512, seed: int = C.RANDOM_SEED) -> str:
    """Write n PNGs to data/images and a labels.csv; return the CSV path."""
    rng = np.random.default_rng(seed)
    C.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rows = [("filename", "label", "bbox")]
    for i in range(n):
        abnormal = rng.random() < abnormal_frac
        arr, bbox = generate_image(abnormal, size=size, seed=int(rng.integers(1 << 31)))
        fname = f"xray_{i:05d}.png"
        Image.fromarray(arr, mode="L").save(C.IMAGES_DIR / fname)
        rows.append((fname, "abnormal" if abnormal else "normal",
                     "" if bbox is None else "|".join(map(str, bbox))))
    with open(C.LABELS_CSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return str(C.LABELS_CSV)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate synthetic chest X-rays.")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--abnormal-frac", type=float, default=0.5)
    p.add_argument("--size", type=int, default=512)
    a = p.parse_args()
    path = generate_dataset(a.n, a.abnormal_frac, a.size)
    print(f"wrote {a.n} images -> {C.IMAGES_DIR} and labels -> {path}")
