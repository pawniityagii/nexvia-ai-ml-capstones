"""Experiment tracking shim.

Uses MLflow when it is installed (the required stack), and falls back to a
local JSON log when it is not, so the training pipeline always runs. The public
surface mirrors the MLflow calls the trainers need.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager

from . import config as C

try:
    import mlflow                                    # noqa: F401
    HAS_MLFLOW = True
except Exception:                                    # noqa: BLE001
    HAS_MLFLOW = False


@contextmanager
def start_run(run_name: str):
    if HAS_MLFLOW:
        import mlflow
        mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=run_name):
            yield _MLflowRun()
    else:
        yield _LocalRun(run_name)


class _MLflowRun:
    def log_params(self, d):
        import mlflow
        mlflow.log_params(d)

    def log_metrics(self, d):
        import mlflow
        mlflow.log_metrics({k: float(v) for k, v in d.items()
                            if isinstance(v, (int, float))})

    def log_artifact(self, path):
        import mlflow
        mlflow.log_artifact(str(path))

    def log_xgboost(self, model):
        import mlflow
        try:
            mlflow.xgboost.log_model(model, name="model")
        except Exception:                            # noqa: BLE001
            pass


class _LocalRun:
    """No-op-ish fallback that records to artifacts/mlflow_fallback.json."""
    def __init__(self, run_name: str):
        self.record = {"run_name": run_name, "ts": time.time(),
                       "params": {}, "metrics": {}}

    def log_params(self, d):
        self.record["params"].update(d)

    def log_metrics(self, d):
        self.record["metrics"].update({k: v for k, v in d.items()
                                       if isinstance(v, (int, float))})

    def log_artifact(self, path):
        self.record.setdefault("artifacts", []).append(str(path))

    def log_xgboost(self, model):
        pass

    def __del__(self):
        try:
            p = C.ARTIFACTS_DIR / "mlflow_fallback.json"
            existing = json.loads(p.read_text()) if p.exists() else []
            existing.append(self.record)
            p.write_text(json.dumps(existing, indent=2))
        except Exception:                            # noqa: BLE001
            pass
