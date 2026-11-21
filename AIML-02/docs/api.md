# API Documentation

Base URL (local): `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

All responses carry the education-only disclaimer where relevant. Nothing here
is a diagnosis.

## `GET /health`
```json
{"status": "ok", "backend": "model" | "fallback", "disclaimer": "..."}
```
`backend` is `"model"` when a trained checkpoint is loaded, else `"fallback"`.

## `POST /analyze`  (multipart form, field `file`)
Upload a PNG/JPG/DICOM scan. Runs inference, builds the Grad-CAM/saliency
overlay, stores a pending case.
```json
{
  "case_id": 12,
  "probability": 0.94,
  "label": "abnormal",
  "mode": "model",
  "overlay_base64": "data:image/png;base64,...",
  "disclaimer": "FOR EDUCATION ONLY - NOT FOR CLINICAL USE. ..."
}
```

## `GET /cases?status=pending&limit=100`
List cases, newest first. `status` ∈ `pending | approved | rejected` (optional).

## `GET /cases/{id}`
One case record.

## `GET /cases/{id}/audit`
Immutable audit trail for a case (creation + every decision).
```json
[{"action":"created","actor":"system","note":"...","ts":...},
 {"action":"approve","actor":"radiologist","note":"...","ts":...}]
```

## `POST /cases/{id}/decision`
```json
{"decision": "approve" | "reject", "actor": "radiologist", "note": ""}
```
Updates case status and appends an audit row.

## `GET /stats`
Counters: total, pending, approved, rejected, abnormal.

## Notes
- The Kafka-free Orthanc worker (`src/orthanc/pull_infer_store.py`) uses the same
  `InferenceService`, so PACS-ingested studies and direct uploads score
  identically.
- Overlays are also written to `artifacts/overlay_*.png`.
