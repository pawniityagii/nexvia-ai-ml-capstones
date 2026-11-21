"""Blend XGBoost and GNN scores and report the lift from fusion.

final = w * xgb_proba + (1 - w) * gnn_score

The weight is swept on the validation slice and the best is applied to test.
This is the step the brief calls for: "Blend its score with XGBoost."
Requires a trained XGBoost model (train_xgboost) and GNN scores (train_gnn).

Run:
    python -m src.models.blend
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ..common import config as C
from ..common import data_loader
from ..features import feature_engineering as fe
from . import graph_builder


def _xgb_proba(feats: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(C.XGB_MODEL_PATH)
    return model.predict_proba(fe.to_model_matrix(feats))[:, 1]


def run() -> dict:
    gnn_path = C.ARTIFACTS_DIR / "gnn_scores.parquet"
    if not (C.XGB_MODEL_PATH.exists() and gnn_path.exists()):
        raise FileNotFoundError(
            "Need both artifacts/xgb_model.json and artifacts/gnn_scores.parquet. "
            "Run train_xgboost and train_gnn first.")

    df = data_loader.load_raw()
    feats = fe.engineer(df)
    feats = graph_builder.attach_ring_features(feats)
    pre = fe.fit_encoders(feats)
    feats = fe.apply_encoders(feats, pre)

    feats["_xgb"] = _xgb_proba(feats)
    gnn = pd.read_parquet(gnn_path).set_index(C.ID_COL)["gnn_score"]
    feats["_gnn"] = feats[C.ID_COL].map(gnn).fillna(feats["_xgb"])

    splits = data_loader.time_split(feats)
    va, te = splits["val"], splits["test"]

    best_w, best_ap = 1.0, -1.0
    for w in np.linspace(0, 1, 21):
        ap = average_precision_score(va[C.TARGET], w * va["_xgb"] + (1 - w) * va["_gnn"])
        if ap > best_ap:
            best_ap, best_w = ap, float(w)

    def ap(frame, w):
        return float(average_precision_score(
            frame[C.TARGET], w * frame["_xgb"] + (1 - w) * frame["_gnn"]))

    out = {
        "best_weight_xgb": best_w,
        "test_pr_auc_xgb_only": ap(te, 1.0),
        "test_pr_auc_gnn_only": ap(te, 0.0),
        "test_pr_auc_blended": ap(te, best_w),
    }
    (C.ARTIFACTS_DIR / "blend_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
