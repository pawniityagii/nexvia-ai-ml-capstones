"""Data-drift monitoring with Evidently + a retraining trigger.

Compares a current window of scored transactions against the training
reference sample (saved by train_xgboost). If the share of drifted features
crosses a threshold, the retraining trigger fires - in production this would
enqueue a training job; here it writes a flag file and returns True.

Written against the current Evidently API (evidently>=0.6):
    from evidently import Report, Dataset, DataDefinition
    from evidently.presets import DataDriftPreset
Export/accessor names vary slightly by minor version, so result extraction and
HTML export are wrapped defensively rather than assumed.

Run:
    python -m src.monitoring.drift --current data/synthetic_transactions.csv
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from ..common import config as C

DRIFT_SHARE_THRESHOLD = 0.4          # fraction of drifted columns that triggers retrain
RETRAIN_FLAG = C.ARTIFACTS_DIR / "RETRAIN_REQUIRED.flag"


def _reference() -> pd.DataFrame:
    if not C.REFERENCE_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            "No reference sample. Train first: python -m src.models.train_xgboost")
    return pd.read_parquet(C.REFERENCE_SAMPLE_PATH)


def _to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce any raw/engineered frame to the model feature columns."""
    cols = [c for c in C.MODEL_FEATURES if c in df.columns]
    return df[cols].astype(float) if cols else df.select_dtypes("number")


def _extract_drift_share(result: dict) -> float:
    """Pull 'share of drifted columns' out of Evidently's result dict robustly."""
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("share_of_drifted_columns", "drift_share",
                         "share_of_drifted_features"):
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
                r = walk(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = walk(v)
                if r is not None:
                    return r
        return None
    val = walk(result)
    return float(val) if val is not None else 0.0


def run_drift(current_df: pd.DataFrame) -> dict:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    ref = _to_model_frame(_reference())
    cur = _to_model_frame(current_df)
    common = [c for c in ref.columns if c in cur.columns]
    ref, cur = ref[common], cur[common]

    # Build datasets (new API uses Dataset/DataDefinition; fall back to raw frames)
    try:
        from evidently import Dataset, DataDefinition
        dd = DataDefinition(numerical_columns=common, categorical_columns=[])
        ref_ds = Dataset.from_pandas(ref, data_definition=dd)
        cur_ds = Dataset.from_pandas(cur, data_definition=dd)
    except Exception:                               # older/newer signature
        ref_ds, cur_ds = ref, cur

    report = Report([DataDriftPreset()])
    snapshot = report.run(cur_ds, ref_ds)           # current first, reference second

    result = {}
    for accessor in ("dict", "as_dict"):
        if hasattr(snapshot, accessor):
            try:
                result = getattr(snapshot, accessor)()
                break
            except Exception:
                continue
    share = _extract_drift_share(result)

    # export HTML report (method name varies: save_html / save)
    html_path = C.ARTIFACTS_DIR / "drift_report.html"
    for method in ("save_html", "save"):
        if hasattr(snapshot, method):
            try:
                getattr(snapshot, method)(str(html_path))
                break
            except Exception:
                continue

    retrain = share >= DRIFT_SHARE_THRESHOLD
    if retrain:
        RETRAIN_FLAG.write_text(json.dumps(
            {"reason": "data_drift", "drift_share": share}, indent=2))

    summary = {"n_features": len(common), "drift_share": round(share, 4),
               "threshold": DRIFT_SHARE_THRESHOLD, "retrain_triggered": retrain,
               "html_report": str(html_path) if html_path.exists() else None}
    (C.ARTIFACTS_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import joblib

    from ..common import data_loader
    from ..features import feature_engineering as fe

    p = argparse.ArgumentParser()
    p.add_argument("--current", type=str, default=None,
                   help="CSV of current transactions (defaults to loaded data)")
    a = p.parse_args()

    raw = pd.read_csv(a.current) if a.current else data_loader.load_raw()
    pre = fe.Preprocessor.from_dict(joblib.load(C.PREPROCESSOR_PATH))
    cur_feats = fe.transform(raw, pre)
    print(json.dumps(run_drift(cur_feats), indent=2))
