# Nexvia AI/ML Internship — Capstone Projects

**Pawni Tyagi · AI / Machine Learning Intern, Nexvia Technologies (Remote) · Oct–Nov 2026**
Reference NEX-INT-2026-0536

This repository holds both allotted capstone projects for the Nexvia AI/ML
internship. Each lives in its own subfolder with complete source code, tests,
documentation and screenshots.

| Project | Title | Folder | Stack |
|---|---|---|---|
| **AIML-01** | Real-Time Financial Fraud Detection Platform | [`AIML-01/`](AIML-01/) | Kafka · XGBoost · PyTorch Geometric · Feast · MLflow · FastAPI · Kubernetes · Evidently |
| **AIML-02** | AI-Assisted Medical Imaging & Diagnosis System | [`AIML-02/`](AIML-02/) | MONAI · PyTorch · Orthanc DICOM · Grad-CAM · DVC · Docker · React |

Each subfolder has its own detailed `README.md` (architecture, install/run steps,
feature breakdown), a `docs/` folder (architecture diagram, model card, API
reference), a `tests/` suite, and a `screenshots/` folder.

---

## AIML-01 — Real-Time Fraud Detection

Ingests a stream of transactions, scores each for fraud risk in milliseconds,
links suspicious accounts into fraud rings with a graph neural network, and gives
analysts a dashboard to review and label cases. Leak-free feature engineering,
PR-AUC evaluation for the <1% fraud rate, Kafka streaming, a Feast feature store,
MLflow tracking, Evidently drift monitoring, and Docker/Kubernetes deployment.

Quickstart (no dataset, no GPU):
```bash
cd AIML-01
pip install -r requirements.txt
python -m src.models.train_xgboost --synthetic
python scripts/export_ring_cache.py
python scripts/replay_demo.py --limit 5000
streamlit run dashboard/app.py
```

## AIML-02 — AI-Assisted Medical Imaging

> **FOR EDUCATION ONLY — NOT FOR CLINICAL USE.** Outputs are model estimates that
> must be reviewed by a qualified radiologist; this system does not provide a
> medical diagnosis.

Uploads chest X-rays (PNG/JPG/DICOM), classifies them with a DenseNet121 transfer
model, explains each prediction with Grad-CAM heatmaps, and routes them to a React
radiologist console where every Approve/Reject decision is written to an audit
log. Includes an Orthanc DICOM-server ingestion worker, a MONAI 3D U-Net
segmentation path, DVC data versioning, and Docker deployment.

Quickstart (no dataset, no GPU):
```bash
cd AIML-02
pip install -r requirements.txt
python scripts/generate_synthetic.py --n 400 --as-dicom 20
python scripts/demo_seed.py --limit 40
uvicorn src.serving.app:app --port 8000      # in one terminal
cd viewer && npm install && npm run dev       # in another
```

---

## Project status (honest)

Both projects are complete and run end to end. The committed demos run on
**synthetic data** so a reviewer can see everything working with no dataset
download and no GPU — the `screenshots/` in each folder are from those runs.
Synthetic data is separable by construction, so its metrics are deliberately
**not** reported as results. The deep-learning models (the PyTorch Geometric GNN
in AIML-01; DenseNet121, Grad-CAM and the MONAI 3D U-Net in AIML-02) are written
to their current APIs and train on Colab/GPU with the real datasets, following
the notebooks in each project. Real-data metrics then populate each README's
results table.

## Internship context

Work produced under the Nexvia Technologies AI/ML internship (NEX-INT-2026-0536).
Published publicly per the internship's own submission requirement.
