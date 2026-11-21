"""Inference orchestration: bytes/array -> prediction + overlay + stored case."""
from __future__ import annotations

import numpy as np

from ..common import config as C
from ..models.predictor import Predictor
from . import case_store, overlay


class InferenceService:
    def __init__(self):
        self.predictor = Predictor()
        self.db = case_store.connect()

    def analyze(self, gray01: np.ndarray, *, filename: str,
                patient_id: str = "", modality: str = "") -> dict:
        pred = self.predictor.predict(gray01)
        png = overlay.overlay_png(gray01, pred["heatmap"])
        overlay_file = C.ARTIFACTS_DIR / f"overlay_{filename}.png"
        overlay_file.write_bytes(png)
        case_id = case_store.add_case(
            self.db, filename=filename, patient_id=patient_id, modality=modality,
            probability=pred["probability"], label=pred["label"],
            model_mode=pred["mode"], overlay_path=str(overlay_file))
        return {
            "case_id": case_id,
            "probability": round(pred["probability"], 4),
            "label": pred["label"],
            "mode": pred["mode"],
            "overlay_base64": overlay.to_base64_png(png),
            "disclaimer": C.DISCLAIMER,
        }
