"""Unified predictor used by the serving layer.

Selects the real DenseNet + Grad-CAM when torch and a trained checkpoint are
present; otherwise uses the deterministic fallback so the app always runs.
Both backends return the same dict:
    {"probability": float, "label": str, "heatmap": HxW float[0,1], "mode": str}
"""
from __future__ import annotations

import numpy as np

from ..common import config as C
from ..common import preprocessing as pp


class Predictor:
    def __init__(self):
        self.backend = "fallback"
        self.model = None
        self.gradcam = None
        try:
            import torch  # noqa: F401
            if C.CLASSIFIER_PATH.exists():
                from .classifier import load_model
                from .gradcam import GradCAM
                self.model = load_model()
                self.gradcam = GradCAM(self.model)
                self.backend = "model"
        except Exception:
            self.backend = "fallback"

    def predict(self, gray01: np.ndarray) -> dict:
        if self.backend != "model":
            from . import fallback
            return fallback.predict(gray01)

        import torch
        x = torch.from_numpy(pp.to_model_input(gray01)).unsqueeze(0).float()
        with torch.no_grad():
            prob = torch.softmax(self.model(x), dim=1)[0, 1].item()
        heatmap = self.gradcam(x, class_idx=1)          # needs grad; own forward
        label = C.CLASSES[1] if prob >= C.DEFAULT_THRESHOLD else C.CLASSES[0]
        return {"probability": float(prob), "label": label,
                "heatmap": heatmap, "mode": "model"}
