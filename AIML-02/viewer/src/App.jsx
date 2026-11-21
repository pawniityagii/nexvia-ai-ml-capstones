import React, { useEffect, useState } from "react";
import { analyze, listCases, decide, getHealth } from "./api.js";
import DisclaimerBanner from "./components/DisclaimerBanner.jsx";
import UploadPanel from "./components/UploadPanel.jsx";
import ScanViewer from "./components/ScanViewer.jsx";
import CaseQueue from "./components/CaseQueue.jsx";

export default function App() {
  const [health, setHealth] = useState(null);
  const [result, setResult] = useState(null);
  const [cases, setCases] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = async () => setCases(await listCases());

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "offline" }));
    refresh().catch(() => {});
  }, []);

  const onAnalyze = async (file) => {
    setBusy(true);
    setError("");
    try {
      setResult(await analyze(file));
      await refresh();
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const onDecision = async (caseId, decision) => {
    await decide(caseId, decision, "");
    await refresh();
    if (result && result.case_id === caseId) {
      setResult({ ...result, status: decision === "approve" ? "approved" : "rejected" });
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <h1>Radiology AI Assist</h1>
        <span className={`badge ${health?.backend === "model" ? "live" : "demo"}`}>
          {health?.backend === "model" ? "LIVE MODEL" : "DEMO (heuristic)"}
        </span>
      </header>
      <DisclaimerBanner text={health?.disclaimer} />
      {error && <div className="error">{error}</div>}
      <div className="grid">
        <section className="left">
          <UploadPanel onAnalyze={onAnalyze} busy={busy} />
          {result && <ScanViewer result={result} onDecision={onDecision} />}
        </section>
        <section className="right">
          <CaseQueue cases={cases} onRefresh={refresh} onDecision={onDecision} />
        </section>
      </div>
    </div>
  );
}
