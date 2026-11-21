"""Central configuration: paths, image spec, class labels.

Chest X-ray binary classification (normal vs abnormal) is the primary task.
The segmentation task (MONAI 3D U-Net) has its own config block and trains on a
GPU via the training notebook.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # repo root (aiml02/)
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"                 # png/jpg chest x-rays
DICOM_DIR = DATA_DIR / "dicom"                   # DICOM studies
LABELS_CSV = DATA_DIR / "labels.csv"             # filename,label
ARTIFACTS_DIR = ROOT / "artifacts"

CLASSIFIER_PATH = ARTIFACTS_DIR / "densenet_chestxray.pt"
CLASSIFIER_META = ARTIFACTS_DIR / "classifier_meta.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SEG_MODEL_PATH = ARTIFACTS_DIR / "unet3d.pt"

for _d in (DATA_DIR, IMAGES_DIR, DICOM_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- classification task ---
CLASSES = ["normal", "abnormal"]                 # index 1 = finding present
IMG_SIZE = 224                                    # DenseNet121 input
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_THRESHOLD = 0.5
RANDOM_SEED = 42

# Grad-CAM target layer for DenseNet121 (last conv feature block).
GRADCAM_TARGET_LAYER = "features.denseblock4.denselayer16.conv2"

# --- segmentation task (MONAI 3D U-Net) ---
SEG_ROI_SIZE = (96, 96, 96)
SEG_IN_CHANNELS = 1
SEG_OUT_CHANNELS = 2                              # background + organ

# --- serving ---
ORTHANC_URL = os.environ.get("ORTHANC_URL", "http://localhost:8042")
ORTHANC_USER = os.environ.get("ORTHANC_USER", "orthanc")
ORTHANC_PASSWORD = os.environ.get("ORTHANC_PASSWORD", "orthanc")

DISCLAIMER = ("FOR EDUCATION ONLY - NOT FOR CLINICAL USE. "
              "Outputs are model estimates and must be reviewed by a qualified "
              "radiologist. This system does not provide a medical diagnosis.")
