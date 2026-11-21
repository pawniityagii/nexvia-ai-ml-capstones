# AI-Assisted Medical Imaging & Diagnosis System

**Nexvia Technologies AI/ML Internship — Project AIML-02**

A platform where chest X-ray scans (PNG/JPG/DICOM) are uploaded, analysed by a
deep-learning model, shown with Grad-CAM explanation heatmaps, and reviewed by a
radiologist who approves or rejects each AI finding — with every decision logged
to an audit trail.

> ## ⚠️ FOR EDUCATION ONLY — NOT FOR CLINICAL USE
> Outputs are model estimates, not a medical diagnosis. Every result requires
> review by a qualified radiologist. Do not use on real patients or real
> clinical decisions. See `docs/model_card.md` for limits and bias.

---

## What it does

| Capability | How |
|---|---|
| Upload scans | PNG/JPG **and DICOM**, parsed with pydicom |
| Classify | DenseNet121 transfer learning (normal vs abnormal) |
| Explain | **Grad-CAM** heatmaps showing which region drove the prediction |
| Segment (extension) | MONAI **3D U-Net** on Medical Segmentation Decathlon |
| Radiologist review | React viewer: overlay toggle, confidence, Approve/Reject |
| Audit | Every decision written to an immutable audit log |
| Ingest from PACS | **Orthanc** DICOM server + pull/infer/store worker |
| Reproducibility | **DVC** data+model versioning, MLflow tracking |
| Deploy | **Docker** Compose (backend + Orthanc + viewer) |

## Required stack → where it lives

| Technology | Location |
|---|---|
| **MONAI** (3D U-Net) | `src/models/segmentation.py`, `src/training/train_segmentation.py` |
| **PyTorch** | classifier + segmentation + Grad-CAM |
| **Grad-CAM** | `src/models/gradcam.py` |
| **Orthanc DICOM server** | `src/orthanc/` + `docker-compose.yml` |
| **DVC** | `dvc.yaml`, `.dvcignore` |
| **Docker** | `Dockerfile`, `viewer/Dockerfile`, `docker-compose.yml` |
| **React** | `viewer/` (Vite) |

---

## Quickstart (no dataset, no GPU — runs on the fallback)

The full app runs on generated synthetic chest X-rays with a deterministic
fallback predictor, so you can demo upload → heatmap → review → audit in a
minute. The real DenseNet plugs in once trained.

```bash
pip install -r requirements.txt

# 1. generate synthetic scans (+ a few DICOM studies) and seed demo cases
python scripts/generate_synthetic.py --n 400 --as-dicom 20
python scripts/demo_seed.py --limit 40

# 2. start the backend  (shows "DEMO (heuristic)" until a model is trained)
uvicorn src.serving.app:app --port 8000

# 3. start the React viewer
cd viewer && npm install && npm run dev      # http://localhost:5173
```

Try the API directly:
```bash
curl -F "file=@data/images/xray_00000.png" localhost:8000/analyze
```

## Train the real model (Colab/GPU)

The DenseNet classifier, Grad-CAM and the 3D U-Net need torch/torchvision/MONAI.

1. Get a real dataset (RSNA Pneumonia or NIH ChestX-ray14 — see `data/README.md`)
   and build `data/labels.csv`, **or** use the synthetic set to smoke-test.
2. `notebooks/train_classifier_colab.ipynb` → trains DenseNet121, saves
   `artifacts/densenet_chestxray.pt`. The backend then auto-switches to
   **LIVE MODEL** and serves real Grad-CAM.
3. `notebooks/train_segmentation_colab.ipynb` → trains the MONAI 3D U-Net on the
   Medical Segmentation Decathlon (the optional segmentation deliverable).

## Full stack with Orthanc (Docker)

```bash
docker compose up -d --build        # orthanc + backend + worker + viewer
python scripts/generate_synthetic.py --as-dicom 20
python scripts/push_to_orthanc.py   # push DICOM studies into Orthanc
# the worker pulls new studies, scores them, and they appear in the viewer
```

---

## Results

The platform is complete and runs end to end; the screenshots below are from the
included synthetic-data demo (the deterministic fallback predictor, badged
`DEMO` in the UI). **Synthetic data is separable by construction, so its numbers
are not reported as results.** Reportable numbers come from training DenseNet121
on a real dataset (RSNA Pneumonia / NIH ChestX-ray14) via
`notebooks/train_classifier_colab.ipynb`, which writes `artifacts/metrics.json`
and flips the backend badge to `LIVE MODEL`:

| Model | Metric | Value |
|---|---|---|
| DenseNet121 classifier | ROC-AUC | _pending real-data run_ |
| | Sensitivity / Specificity | _pending real-data run_ |
| 3D U-Net (segmentation) | Dice | _pending Decathlon run_ |

## Screenshots

From the synthetic-data demo (`screenshots/`):

| | |
|---|---|
| Classifier + Grad-CAM/explanation overlays | `screenshots/01_gradcam_overlays.png` |
| React radiologist viewer (home) | `screenshots/02_viewer_home.png` |
| Viewer: analysed scan, heatmap, Approve/Reject, case queue | `screenshots/03_viewer_result.png` |
| FastAPI backend (`/docs`) | `screenshots/04_fastapi_docs.png` |

![Radiologist viewer with heatmap overlay and case queue](screenshots/03_viewer_result.png)
![Explanation heatmaps](screenshots/01_gradcam_overlays.png)

## Verification status (honest)

- **Runtime-tested here:** synthetic X-ray generation, DICOM read/write,
  preprocessing, the fallback predictor (cleanly separates normal/abnormal and
  the heatmap localises the opacity), overlay rendering, the full FastAPI backend
  (upload → analyze → decision → audit) via TestClient, the pytest suite
  (7 tests), and the **React viewer builds** with Vite.
- **Written to current APIs, run in your environment / Colab:** the DenseNet
  classifier + Grad-CAM, the MONAI 3D U-Net, the Orthanc worker (needs a running
  Orthanc), and DVC/Docker (configs validated).

## Project structure

```
AIML-02/
├── src/
│   ├── common/      config, synthetic X-rays, DICOM I/O, preprocessing
│   ├── models/      densenet classifier, grad-cam, 3D U-Net, fallback, predictor
│   ├── training/    train_classifier, train_segmentation
│   ├── serving/     FastAPI app, inference, case store + audit, overlay
│   └── orthanc/     DICOM server config + pull/infer/store worker
├── viewer/          React (Vite) radiologist viewer
├── scripts/         data generation, demo seed, Orthanc push
├── notebooks/       EDA + Colab training (classifier, segmentation)
├── tests/           pytest suite
├── docs/            architecture, model card, API, presentation, reflection
├── dvc.yaml         DVC data/model pipeline
└── docker-compose.yml
```

See `docs/` for the architecture diagram, model card (limits & bias), and API
reference.
