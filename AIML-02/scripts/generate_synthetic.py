"""Generate synthetic chest X-rays (PNG + optional DICOM) for the demo."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config as C
from src.common import synthetic
from src.common.dicom_io import array_to_dicom
from src.common.preprocessing import load_image


def main(n: int, as_dicom: int):
    synthetic.generate_dataset(n=n)
    print(f"wrote {n} PNGs -> {C.IMAGES_DIR}")
    if as_dicom:
        C.DICOM_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(C.IMAGES_DIR.glob("*.png"))[:as_dicom]
        for i, png in enumerate(files):
            arr = (load_image(png) * 255).astype("uint8")
            array_to_dicom(arr, C.DICOM_DIR / f"study_{i:04d}.dcm",
                           patient_id=f"SYN{i:04d}")
        print(f"wrote {len(files)} DICOM studies -> {C.DICOM_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--as-dicom", type=int, default=20,
                   help="how many to also wrap as DICOM")
    a = p.parse_args()
    main(a.n, a.as_dicom)
