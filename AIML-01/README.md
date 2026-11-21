# Real-Time Financial Fraud Detection Platform

**Nexvia Technologies AI/ML Internship — Project AIML-01**

An end-to-end fraud platform that ingests a stream of transactions, scores each
one for fraud risk in milliseconds, links suspicious accounts into fraud rings
with a graph neural network, and gives analysts a dashboard to review cases.

> **Education / portfolio project. Not a production or regulated fraud system.**
> Scores are model estimates, not financial determinations. The "real-time"
> path runs on a single node; the fraud-ring model is trained offline.

---

## What it does

| Capability | How |
|---|---|
| Score each transaction in ms | FastAPI service + online feature store, p50 ≈ 2 ms (measured on synthetic replay) |
| Handle <1% fraud honestly | PR-AUC (not accuracy), class weighting / SMOTE, PR-curve threshold tuning |
| Detect fraud **rings** | Graph of shared card/device links + GraphSAGE GNN (PyTorch Geometric) |
| Blend signals | Weighted fusion of XGBoost + GNN scores |
| Train/serve parity | One feature definition shared via Feast |
| Catch model decay | Evidently drift report + automatic retraining trigger |
| Review queue | Streamlit analyst console with case status + feedback labels |
| Reproducibility | MLflow run tracking |
| Deploy | Docker Compose (local) and Kubernetes/Minikube manifests |

## Required stack → where it lives

| Technology | Location |
|---|---|
| **XGBoost** | `src/models/train_xgboost.py` |
| **PyTorch Geometric** (GNN) | `src/models/train_gnn.py`, `src/models/graph_builder.py` |
| **Feast** (feature store) | `feature_repo/`, `scripts/export_feast_source.py` |
| **MLflow** (tracking) | `src/common/tracking.py` (used by both trainers) |
| **Kafka** (streaming) | `src/streaming/producer.py`, `consumer.py` |
| **FastAPI** (serving) | `src/serving/app.py`, `scorer.py` |
| **Evidently AI** (drift) | `src/monitoring/drift.py` |
| **Kubernetes** | `deploy/k8s/` |

---

## Quickstart (no dataset, no Kafka, no GPU)

The whole pipeline runs on generated synthetic data so you can see it work in
one minute, then swap in the real dataset.

```bash
pip install -r requirements.txt

# 1. generate synthetic data + train XGBoost (writes artifacts/, metrics.json)
python -m src.models.train_xgboost --synthetic

# 2. cache graph features for the online path
python scripts/export_ring_cache.py

# 3. replay transactions through the scorer (no Kafka needed) -> fills case store
python scripts/replay_demo.py --limit 5000

# 4. open the analyst dashboard
streamlit run dashboard/app.py
```

Run the scoring API on its own:
```bash
uvicorn src.serving.app:app --port 8000
curl -X POST localhost:8000/score -H 'Content-Type: application/json' \
  -d '{"TransactionID":1,"TransactionDT":500000,"TransactionAmt":2500,"card4":"visa","DeviceInfo":"new-device#1","P_emaildomain":"gmail.com"}'
```

## Run on the real dataset

1. Download **IEEE-CIS Fraud Detection** from Kaggle (see `data/README.md`).
2. Put `train_transaction.csv` (and `train_identity.csv`) in `data/`.
3. Re-run training — it uses the real files automatically:
   ```bash
   python -m src.models.train_xgboost     # no --synthetic flag
   ```
4. Train the GNN on Colab/GPU (see `notebooks/gnn_colab.ipynb`), then:
   ```bash
   python -m src.models.blend             # fuse XGBoost + GNN, report lift
   ```

## Full streaming + MLOps stack

```bash
docker compose up -d                      # Kafka, MLflow, API, dashboard
python -m src.streaming.producer --rate 50     # replay as a live stream
python -m src.streaming.consumer --from-beginning  # score the stream
python -m src.monitoring.drift            # drift report + retrain trigger
```
Feature store:
```bash
python scripts/export_feast_source.py
cd feature_repo && feast apply && feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
```

---

## Results

The pipeline is complete and runs end to end; the screenshots below are from the
included synthetic-data run. **Synthetic data is separable by construction, so
its scores are near-perfect and are deliberately not reported as results** - they
only prove the pipeline works. Reportable numbers come from training on the real
**IEEE-CIS** dataset (`python -m src.models.train_xgboost`, then the GNN on
Colab), which writes `artifacts/metrics.json`:

| Model | PR-AUC (test) | ROC-AUC (test) |
|---|---|---|
| XGBoost | _pending IEEE-CIS run_ | _pending IEEE-CIS run_ |
| GNN (GraphSAGE) | _pending IEEE-CIS run_ | _pending IEEE-CIS run_ |
| Blended | _pending IEEE-CIS run_ | _pending IEEE-CIS run_ |

## Screenshots

From the synthetic-data run (`screenshots/`):

| | |
|---|---|
| Test PR / ROC curves | `screenshots/01_pr_roc_curves.png` |
| Confusion matrix | `screenshots/02_confusion_matrix.png` |
| Top feature importances | `screenshots/03_feature_importance.png` |
| Fraud-ring graph (shared card/device links) | `screenshots/04_fraud_ring_graph.png` |
| FastAPI scoring service (`/docs`) | `screenshots/05_fastapi_docs.png` |
| Streamlit analyst dashboard | `screenshots/06_analyst_dashboard.png` |

![PR and ROC curves](screenshots/01_pr_roc_curves.png)
![Analyst dashboard](screenshots/06_analyst_dashboard.png)

## Verification status (honest)

- **Runtime-tested here:** synthetic data, feature engineering (leak-free),
  graph + ring features, XGBoost training + MLflow logging, online scorer
  (p50 ≈ 2 ms), SQLite case store, FastAPI endpoints, the pytest suite (6 tests,
  incl. a leakage guard).
- **Written to the documented current API, run in your environment / Colab:**
  the GNN and blend (need torch + PyTorch Geometric), Feast, Evidently, and the
  Kafka producer/consumer (need a broker). The Kubernetes manifests validate
  with `kubectl --dry-run`.

## Project structure

```
AIML-01/
├── src/
│   ├── common/      config, synthetic data, loader, MLflow shim
│   ├── features/    leak-free feature engineering
│   ├── models/      xgboost, graph builder, GNN, blend
│   ├── streaming/   kafka producer + consumer
│   ├── serving/     FastAPI app, scorer, online features, case store
│   └── monitoring/  evidently drift + retrain trigger
├── feature_repo/    Feast feature store
├── dashboard/       Streamlit analyst console
├── deploy/k8s/      Kubernetes manifests
├── notebooks/       EDA + Colab GNN training
├── scripts/         demo replay, cache + feast exporters
├── tests/           pytest suite
└── docs/            architecture, model card, API, presentation, reflection
```

See `docs/` for the architecture diagram, model card (limits & bias), and API
documentation.
