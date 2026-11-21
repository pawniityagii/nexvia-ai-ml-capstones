"""FastAPI backend for the medical-imaging diagnosis-assist platform.

Endpoints:
  POST /analyze                 upload a PNG/JPG/DICOM scan -> prediction + overlay
  GET  /cases                   list cases (optionally by status)
  GET  /cases/{id}              one case
  GET  /cases/{id}/audit        audit trail for a case
  POST /cases/{id}/decision     radiologist approve/reject (writes audit)
  GET  /stats                   dashboard counters
  GET  /health, GET /disclaimer

Every prediction is an estimate for a qualified radiologist to review; the
service never issues a diagnosis. The React viewer calls these endpoints.

Run: uvicorn src.serving.app:app --port 8000
"""
from __future__ import annotations

import io
import tempfile
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from ..common import config as C
from ..common import dicom_io
from . import case_store
from .inference import InferenceService

_STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _STATE["svc"] = InferenceService()
    yield
    _STATE.clear()


app = FastAPI(title="Medical Imaging Diagnosis Assist", version="1.0.0",
              lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Decision(BaseModel):
    decision: str            # "approve" | "reject"
    actor: str = "radiologist"
    note: str = ""


def _load_upload(raw: bytes, filename: str) -> np.ndarray:
    if filename.lower().endswith((".dcm", ".dicom")) or raw[128:132] == b"DICM":
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
            f.write(raw)
            tmp = f.name
        return dicom_io.read_dicom(tmp)
    img = Image.open(io.BytesIO(raw)).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


@app.get("/health")
def health():
    svc: InferenceService = _STATE.get("svc")
    return {"status": "ok",
            "backend": svc.predictor.backend if svc else None,
            "disclaimer": C.DISCLAIMER}


@app.get("/disclaimer")
def disclaimer():
    return {"disclaimer": C.DISCLAIMER}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        gray = _load_upload(raw, file.filename or "upload")
    except Exception as e:                           # noqa: BLE001
        raise HTTPException(400, f"could not read image: {e}")
    patient_id, modality = "", ""
    if (file.filename or "").lower().endswith((".dcm", ".dicom")):
        try:
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
                f.write(raw)
                tags = dicom_io.dicom_tags(f.name)
            patient_id, modality = tags["patient_id"], tags["modality"]
        except Exception:                            # noqa: BLE001
            pass
    return _STATE["svc"].analyze(gray, filename=file.filename or "upload",
                                 patient_id=patient_id, modality=modality)


@app.get("/cases")
def cases(status: str | None = None, limit: int = 100):
    return case_store.list_cases(_STATE["svc"].db, status=status, limit=limit)


@app.get("/stats")
def stats():
    return case_store.stats(_STATE["svc"].db)


@app.get("/cases/{case_id}")
def case(case_id: int):
    c = case_store.get_case(_STATE["svc"].db, case_id)
    if not c:
        raise HTTPException(404, "case not found")
    return c


@app.get("/cases/{case_id}/audit")
def audit(case_id: int):
    return case_store.list_audit(_STATE["svc"].db, case_id)


@app.post("/cases/{case_id}/decision")
def decide(case_id: int, body: Decision):
    if not case_store.get_case(_STATE["svc"].db, case_id):
        raise HTTPException(404, "case not found")
    try:
        status = case_store.set_decision(
            _STATE["svc"].db, case_id, body.decision, body.actor, body.note)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"case_id": case_id, "status": status}


# Serve the built React viewer when it is present, so a single deployment hosts
# both the UI and the API at one origin. The API routes above are registered
# first, so they take precedence; everything else is served from the SPA build.
_VIEWER_DIST = C.ROOT / "viewer" / "dist"
if _VIEWER_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_VIEWER_DIST), html=True), name="viewer")
