import React, { useState } from "react";

export default function ScanViewer({ result, onDecision }) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const pct = (result.probability * 100).toFixed(1);
  const abnormal = result.label === "abnormal";
  return (
    <div className="panel viewer">
      <h2>Result - case #{result.case_id}</h2>
      <div className="image-wrap">
        {showHeatmap ? (
          <img src={result.overlay_base64} alt="AI heatmap overlay" />
        ) : (
          <img src={result.overlay_base64} alt="scan" style={{ filter: "grayscale(1)" }} />
        )}
      </div>
      <label className="toggle">
        <input
          type="checkbox"
          checked={showHeatmap}
          onChange={(e) => setShowHeatmap(e.target.checked)}
        />
        Show AI heatmap
      </label>

      <div className={`verdict ${abnormal ? "abnormal" : "normal"}`}>
        {result.label.toUpperCase()}
        <span className="conf">confidence {pct}%</span>
      </div>
      <div className="confbar">
        <div className="fill" style={{ width: `${pct}%` }} />
      </div>
      {result.mode !== "model" && (
        <p className="hint">Heuristic demo output - not a trained model.</p>
      )}

      <div className="actions">
        <button className="approve" onClick={() => onDecision(result.case_id, "approve")}>
          Approve (confirm finding)
        </button>
        <button className="reject" onClick={() => onDecision(result.case_id, "reject")}>
          Reject (overrule AI)
        </button>
      </div>
      {result.status && <p className="status">Radiologist decision: {result.status}</p>}
    </div>
  );
}
