// In production the viewer is served by the FastAPI backend itself, so API
// calls go to the same origin (BASE = ""). For local `npm run dev`, Vite proxies
// the API paths to the backend on :8000 (see vite.config.js), so same-origin
// paths work there too. Override with VITE_API_BASE if the API is elsewhere.
const BASE = import.meta.env.VITE_API_BASE || "";

export async function analyze(file) {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${BASE}/analyze`, { method: "POST", body: form });
  if (!r.ok) throw new Error(`analyze failed: ${r.status}`);
  return r.json();
}

export async function listCases(status) {
  const q = status ? `?status=${status}` : "";
  const r = await fetch(`${BASE}/cases${q}`);
  return r.json();
}

export async function getAudit(caseId) {
  const r = await fetch(`${BASE}/cases/${caseId}/audit`);
  return r.json();
}

export async function decide(caseId, decision, note) {
  const r = await fetch(`${BASE}/cases/${caseId}/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision, actor: "radiologist", note }),
  });
  if (!r.ok) throw new Error(`decision failed: ${r.status}`);
  return r.json();
}

export async function getStats() {
  const r = await fetch(`${BASE}/stats`);
  return r.json();
}

export async function getHealth() {
  const r = await fetch(`${BASE}/health`);
  return r.json();
}
