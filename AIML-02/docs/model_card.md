# Model Card - AI-Assisted Medical Imaging

## ⚠️ FOR EDUCATION ONLY - NOT FOR CLINICAL USE
This system does not provide a medical diagnosis. Outputs are model estimates
that must be reviewed by a qualified radiologist. It must not be used on real
patients or to inform real clinical decisions.

## Intended use
- Internship / portfolio demonstration of a medical-imaging ML pipeline with
  explainability and a human-in-the-loop review workflow.

## Out of scope
- Any real diagnosis, triage, or patient-facing use. A real deployment would
  require regulatory clearance (e.g. FDA/CE), prospective clinical validation,
  bias auditing across demographics, and integration controls far beyond this
  project.

## Models
- **Classifier:** DenseNet121 (ImageNet-pretrained), final layer replaced for
  binary normal/abnormal; explained with Grad-CAM.
- **Segmentation (extension):** MONAI 3D U-Net, Dice loss, trained on the
  Medical Segmentation Decathlon.
- **Fallback:** a deterministic opacity-detection heuristic for the zero-GPU
  demo. Not a neural network, not diagnostic.

## Data
- Real: RSNA Pneumonia / NIH ChestX-ray14 / CheXpert (see `data/README.md`).
- Synthetic: procedurally drawn pseudo-X-rays for development/demo only.
  Synthetic metrics are near-perfect by construction and never reported.

## Evaluation
- Classification reports **ROC-AUC, sensitivity and specificity** (not just
  accuracy) - sensitivity matters most for screening, where a missed finding is
  the costly error. Segmentation reports the **Dice** coefficient.

## Limitations
- Public chest-X-ray labels are noisy and often derived from reports, not pixel
  truth; models learn dataset artefacts (view, device, text markers).
- Binary normal/abnormal collapses many distinct pathologies into one label.
- Grad-CAM is a coarse localisation, not a segmentation or a clinical
  explanation; it can highlight spurious correlates.
- The fallback heuristic only reacts to bright opacities; it is a placeholder.

## Bias & fairness
- Chest-X-ray datasets are known to carry demographic and site biases (scanner,
  hospital, patient mix). Model performance can differ by sex, age and source
  institution. Any real use demands stratified performance reporting and a bias
  audit. Flag-rate parity across groups should be checked, not assumed.
- A false negative (missed finding) and a false positive (false alarm) carry
  very different clinical costs; the operating threshold must be set with
  clinicians, not left at 0.5.

## Human oversight
Every prediction is a *pending* case until a radiologist approves or rejects it;
all decisions are logged. The system supports a clinician; it never acts alone.
