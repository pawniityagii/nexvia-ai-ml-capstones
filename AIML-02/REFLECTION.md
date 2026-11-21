# Individual Reflection — AIML-02

> Draft to make your own. Edit it into your voice and keep only what is true for
> you before submitting; a reviewer may ask about anything written here.

## What I built and own
I built the medical-imaging diagnosis-assist platform: DICOM and PNG/JPG handling
with pydicom, preprocessing into a DenseNet121 input, a transfer-learning
classifier with Grad-CAM explanation heatmaps, a FastAPI backend that scores a
scan and returns a heatmap overlay and confidence, and a React radiologist viewer
where each prediction is a pending case a radiologist approves or rejects, with
every decision written to an audit log. An Orthanc worker ingests DICOM studies
from a PACS, and a MONAI 3D U-Net provides the segmentation path.

## What I learned
New to me here: the DICOM format and how to read pixel data correctly (photometric
interpretation, rescale), Grad-CAM and why explainability matters for a medical
tool, MONAI and 3D segmentation, the Orthanc DICOM server, DVC for data
versioning, and containerising a multi-service app with Docker.

## Hardest problem
Making Grad-CAM localise the finding correctly, and designing the system so a
human is always in the loop — the model never decides on its own, every case is
reviewed, and the "education only, not clinical use" boundary is enforced in the
UI, the API and the docs.

## Ethics and safety
A medical model must not act alone. I kept the radiologist approve/reject step and
the audit trail central, and I am clear that chest-X-ray datasets carry
demographic and site bias that a real deployment would have to audit.

## What is real vs next
The app runs end to end on synthetic X-rays with a deterministic fallback
predictor, badged DEMO in the UI, so the numbers are not results. The next step is
training DenseNet121 on a real dataset (RSNA Pneumonia / NIH ChestX-ray14) for
reportable ROC-AUC and sensitivity, after which the backend serves the real model
and real Grad-CAM.
