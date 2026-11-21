"""Poll an Orthanc DICOM server, pull new studies, run inference, store results.

Implements the brief's step 5: "write a service that pulls new studies, runs
inference, and stores results." Polls the Orthanc REST API for new instances,
downloads each DICOM, scores it with the same InferenceService the API uses, and
records a case (with audit entry) in the shared store.

Run (needs Orthanc running; docker-compose up orthanc):
    python -m src.orthanc.pull_infer_store --interval 5
"""
from __future__ import annotations

import argparse
import io
import time

import requests

from ..common import config as C
from ..common import dicom_io
from ..serving.inference import InferenceService


def _auth():
    return (C.ORTHANC_USER, C.ORTHANC_PASSWORD)


def list_instances() -> list[str]:
    r = requests.get(f"{C.ORTHANC_URL}/instances", auth=_auth(), timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_instance(instance_id: str) -> bytes:
    r = requests.get(f"{C.ORTHANC_URL}/instances/{instance_id}/file",
                     auth=_auth(), timeout=30)
    r.raise_for_status()
    return r.content


def poll_once(svc: InferenceService, seen: set) -> int:
    new = 0
    for iid in list_instances():
        if iid in seen:
            continue
        seen.add(iid)
        try:
            raw = fetch_instance(iid)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
                f.write(raw)
                path = f.name
            gray = dicom_io.read_dicom(path)
            tags = dicom_io.dicom_tags(path)
            res = svc.analyze(gray, filename=f"orthanc_{iid}.dcm",
                              patient_id=tags["patient_id"], modality=tags["modality"])
            new += 1
            print(f"scored instance {iid[:12]} -> {res['label']} "
                  f"p={res['probability']:.3f} (case {res['case_id']})")
        except Exception as e:                       # noqa: BLE001
            print(f"skip {iid[:12]}: {e}")
    return new


def main(interval: float, once: bool):
    svc = InferenceService()
    seen: set = set()
    print(f"polling {C.ORTHANC_URL} every {interval}s ...")
    while True:
        try:
            n = poll_once(svc, seen)
            if n:
                print(f"processed {n} new instance(s)")
        except requests.RequestException as e:
            print(f"orthanc unreachable: {e}")
        if once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=5)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    main(a.interval, a.once)
