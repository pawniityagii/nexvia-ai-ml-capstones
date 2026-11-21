# 10-Minute Presentation Outline

One idea per slide, ~1 minute each.

1. **Title (0:30)** — AI-Assisted Medical Imaging & Diagnosis System. Your name,
   Nexvia AIML-02. One line: "upload a scan, get an explained prediction, a
   radiologist signs off." Lead with the education-only disclaimer.

2. **Problem (1:00)** — Radiologists are overloaded; AI triage can help *if* it
   is explainable and keeps a human in the loop. Three needs: accuracy,
   explanation, and auditable oversight.

3. **Data + DICOM (1:00)** — Chest X-rays (RSNA/NIH). What DICOM is and why it
   matters (headers, photometric interpretation, rescale). Class balance.

4. **Architecture (1:30)** — Walk the diagram: upload/Orthanc → preprocess →
   DenseNet + Grad-CAM → overlay → radiologist review → audit. Note DVC, MLflow,
   Docker, the React viewer.

5. **Model (1:00)** — DenseNet121 transfer learning, why transfer learning for
   small medical sets, metrics that matter (ROC-AUC, **sensitivity** over
   accuracy). Show the real numbers.

6. **Explainability (1:00)** — Grad-CAM: show a heatmap over a finding. Explain
   that you validated localisation against a known opacity in synthetic data.

7. **Demo (1:30)** — Screen-record: upload a scan in the React viewer, toggle
   the heatmap, read the confidence, Approve/Reject, open the audit trail. Then
   show the Orthanc path pulling a study.

8. **Segmentation + MLOps (1:00)** — The MONAI 3D U-Net (Dice), DVC-versioned
   data, MLflow runs, Docker Compose stack.

9. **Ethics, limits, next steps (1:00)** — Education-only, bias in CXR datasets,
   human-in-the-loop by design; next: more pathology classes, real segmentation
   overlay in the viewer, demographic performance audit.

10. **Close (0:30)** — What you built and owned; the one thing you learned.

**Tips:** keep the disclaimer visible; lead with the demo video if a live demo
is risky; name the trade-offs (fallback vs trained model, classification vs
segmentation) — honesty about scope reads as maturity.
