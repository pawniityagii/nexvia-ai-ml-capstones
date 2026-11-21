"""Populate the case store by scoring synthetic scans - no Orthanc, no torch.

Gives the dashboard/viewer something to show for a screen-recorded demo.
Run after generate_synthetic.py (or it will generate a small set itself).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config as C
from src.common import synthetic
from src.common.preprocessing import load_image
from src.serving.inference import InferenceService


def main(limit: int):
    if not C.LABELS_CSV.exists():
        synthetic.generate_dataset(n=max(limit, 60))
    svc = InferenceService()
    files = sorted(C.IMAGES_DIR.glob("*.png"))[:limit]
    flagged = 0
    for png in files:
        gray = load_image(png)
        res = svc.analyze(gray, filename=png.name)
        flagged += res["label"] == "abnormal"
    print(f"seeded {len(files)} cases, {flagged} flagged abnormal "
          f"(backend={svc.predictor.backend})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=40)
    a = p.parse_args()
    main(a.limit)
