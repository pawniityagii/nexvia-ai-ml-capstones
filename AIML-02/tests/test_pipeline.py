"""Tests for the torch-free pipeline: data, DICOM, preprocessing, fallback,
overlay, and the FastAPI backend (upload -> decision -> audit)."""
import io
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from src.common import dicom_io, synthetic
from src.common import preprocessing as pp
from src.models import fallback
from src.serving import overlay


def test_synthetic_abnormal_has_bbox():
    _, bbox_a = synthetic.generate_image(True, size=256, seed=1)
    _, bbox_n = synthetic.generate_image(False, size=256, seed=1)
    assert bbox_a is not None and bbox_n is None


def test_dicom_roundtrip():
    arr, _ = synthetic.generate_image(True, size=128, seed=2)
    d = tempfile.mkdtemp()
    p = dicom_io.array_to_dicom(arr, os.path.join(d, "t.dcm"), patient_id="P9")
    assert dicom_io.is_dicom(p)
    back = dicom_io.read_dicom(p)
    assert back.shape == (128, 128)
    assert 0.0 <= back.min() and back.max() <= 1.0
    assert dicom_io.dicom_tags(p)["patient_id"] == "P9"


def test_preprocessing_shape():
    arr, _ = synthetic.generate_image(False, size=300, seed=3)
    x = pp.to_model_input(arr / 255.0)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32


def test_fallback_separates_classes():
    pn, pa = [], []
    for s in range(20):
        n, _ = synthetic.generate_image(False, 512, seed=s)
        a, _ = synthetic.generate_image(True, 512, seed=500 + s)
        pn.append(fallback.predict(n / 255.0)["probability"])
        pa.append(fallback.predict(a / 255.0)["probability"])
    assert np.mean(pa) > 0.7 and np.mean(pn) < 0.3


def test_fallback_heatmap_localizes():
    arr, bbox = synthetic.generate_image(True, 512, seed=7)
    r = fallback.predict(arr / 255.0)
    hm = r["heatmap"]
    peak = np.unravel_index(hm.argmax(), hm.shape)       # (row, col) in 224 space
    x0, y0, x1, y1 = [v * 224 / 512 for v in bbox]
    # peak should fall near the opacity box (allow a margin)
    assert (x0 - 20) <= peak[1] <= (x1 + 20)
    assert (y0 - 20) <= peak[0] <= (y1 + 20)


def test_overlay_returns_png():
    arr, _ = synthetic.generate_image(True, 256, seed=4)
    r = fallback.predict(arr / 255.0)
    png = overlay.overlay_png(arr / 255.0, r["heatmap"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert overlay.to_base64_png(png).startswith("data:image/png;base64,")


def test_backend_analyze_decision_audit(tmp_path):
    from fastapi.testclient import TestClient
    from src.serving.app import app

    arr, _ = synthetic.generate_image(True, 512, seed=11)
    buf = io.BytesIO()
    Image.fromarray(arr, "L").save(buf, "PNG")
    buf.seek(0)
    with TestClient(app) as c:
        r = c.post("/analyze", files={"file": ("a.png", buf, "image/png")}).json()
        assert "overlay_base64" in r and r["disclaimer"]
        cid = r["case_id"]
        d = c.post(f"/cases/{cid}/decision",
                   json={"decision": "approve", "actor": "dr", "note": "ok"}).json()
        assert d["status"] == "approved"
        audit = c.get(f"/cases/{cid}/audit").json()
        actions = [a["action"] for a in audit]
        assert "created" in actions and "approve" in actions
