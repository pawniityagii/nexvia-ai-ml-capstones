"""Push synthetic DICOM studies into a running Orthanc server (for the demo).

Run after generate_synthetic.py --as-dicom N and `docker compose up orthanc`:
    python scripts/push_to_orthanc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config as C


def main():
    files = sorted(C.DICOM_DIR.glob("*.dcm"))
    if not files:
        print("no DICOM files - run: python scripts/generate_synthetic.py --as-dicom 20")
        return
    ok = 0
    for dcm in files:
        r = requests.post(f"{C.ORTHANC_URL}/instances", data=dcm.read_bytes(),
                          auth=(C.ORTHANC_USER, C.ORTHANC_PASSWORD),
                          headers={"Content-Type": "application/dicom"}, timeout=30)
        if r.ok:
            ok += 1
        else:
            print(f"failed {dcm.name}: {r.status_code}")
    print(f"pushed {ok}/{len(files)} studies to {C.ORTHANC_URL}")


if __name__ == "__main__":
    main()
