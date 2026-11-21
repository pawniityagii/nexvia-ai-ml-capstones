"""Train the XGBoost fraud classifier with MLflow tracking.

Pipeline (honest, leak-free):
  1. Load raw (IEEE-CIS if present, else synthetic).
  2. Engineer features over FULL history (leak-free time/velocity/ring features).
  3. Chronological train/val/test split.
  4. Fit frequency encoders + imputation on the TRAIN slice only.
  5. Train XGBoost with scale_pos_weight (or SMOTE) for the <1% fraud rate.
  6. Evaluate with precision-recall AUC (PR-AUC), not accuracy.
  7. Tune the decision threshold on the validation PR curve.
  8. Log params/metrics/model to MLflow; persist artifacts for serving + drift.

Run:
    python -m src.models.train_xgboost                 # auto data
    python -m src.models.train_xgboost --synthetic     # force synthetic
    python -m src.models.train_xgboost --smote         # SMOTE instead of weights
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score)

from ..common import config as C
from ..common import data_loader
from ..features import feature_engineering as fe
from . import graph_builder


def _prepare() -> dict:
    df = data_loader.load_raw(prefer_synthetic=_ARGS.synthetic)
    feats = fe.engineer(df)                      # leak-free, full history
    feats = graph_builder.attach_ring_features(feats)
    splits = data_loader.time_split(feats)
    pre = fe.fit_encoders(splits["train"])       # fit on train slice only
    for k in splits:
        splits[k] = fe.apply_encoders(splits[k], pre)
    return {"splits": splits, "pre": pre, "source": df.attrs.get("source", "unknown")}


def _matrix(frame: pd.DataFrame):
    X = fe.to_model_matrix(frame)
    y = frame[C.TARGET].astype(int).to_numpy()
    return X, y


def _tune_threshold(y_true, proba) -> float:
    """Pick the threshold maximising F1 on the PR curve."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    # thr has len-1 vs prec/rec; align
    best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[best]) if len(thr) else C.DEFAULT_THRESHOLD


def train() -> dict:
    import xgboost as xgb

    from ..common import tracking

    prep = _prepare()
    splits = prep["splits"]
    Xtr, ytr = _matrix(splits["train"])
    Xva, yva = _matrix(splits["val"])
    Xte, yte = _matrix(splits["test"])

    pos = max(1, int(ytr.sum()))
    neg = int((ytr == 0).sum())
    scale_pos_weight = neg / pos

    params = dict(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, objective="binary:logistic",
        eval_metric="aucpr", tree_method="hist",
        random_state=C.RANDOM_SEED, n_jobs=-1,
    )
    if _ARGS.smote:
        from imblearn.over_sampling import SMOTE
        Xtr, ytr = SMOTE(random_state=C.RANDOM_SEED).fit_resample(Xtr, ytr)
    else:
        params["scale_pos_weight"] = scale_pos_weight

    with tracking.start_run(run_name="xgboost") as run:
        model = xgb.XGBClassifier(**params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        p_va = model.predict_proba(Xva)[:, 1]
        p_te = model.predict_proba(Xte)[:, 1]
        threshold = _tune_threshold(yva, p_va)

        metrics = {
            "data_source": prep["source"],
            "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
            "train_fraud_rate": float(np.mean(splits["train"][C.TARGET])),
            "scale_pos_weight": float(scale_pos_weight),
            "threshold": threshold,
            "val_pr_auc": float(average_precision_score(yva, p_va)),
            "val_roc_auc": float(roc_auc_score(yva, p_va)),
            "test_pr_auc": float(average_precision_score(yte, p_te)),
            "test_roc_auc": float(roc_auc_score(yte, p_te)),
            "test_f1_at_threshold": float(f1_score(yte, (p_te >= threshold).astype(int))),
        }
        run.log_params({k: params[k] for k in
                        ("n_estimators", "max_depth", "learning_rate")})
        run.log_params({"imbalance": "smote" if _ARGS.smote else "scale_pos_weight"})
        run.log_metrics(metrics)

        # persist for serving + drift baseline
        model.save_model(C.XGB_MODEL_PATH)
        joblib.dump(prep["pre"].to_dict(), C.PREPROCESSOR_PATH)
        C.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        C.FEATURE_META_PATH.write_text(json.dumps(
            {"model_features": C.MODEL_FEATURES, "threshold": threshold}, indent=2))
        ref = fe.to_model_matrix(splits["train"]).sample(
            min(5000, len(splits["train"])), random_state=C.RANDOM_SEED)
        ref.to_parquet(C.REFERENCE_SAMPLE_PATH)
        run.log_xgboost(model)
        run.log_artifact(C.XGB_MODEL_PATH)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true", help="force synthetic data")
    p.add_argument("--smote", action="store_true", help="use SMOTE not class weights")
    _ARGS = p.parse_args()
    train()
else:  # importable without CLI args
    _ARGS = argparse.Namespace(synthetic=False, smote=False)
