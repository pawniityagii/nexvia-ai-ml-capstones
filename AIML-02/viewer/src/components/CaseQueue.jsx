import React, { useState } from "react";
import AuditTrail from "./AuditTrail.jsx";

export default function CaseQueue({ cases, onRefresh, onDecision }) {
  const [open, setOpen] = useState(null);
  return (
    <div className="panel">
      <div className="queue-head">
        <h2>Case queue</h2>
        <button className="ghost" onClick={onRefresh}>Refresh</button>
      </div>
      {!cases.length && <p className="hint">No cases yet. Analyze a scan.</p>}
      <table className="cases">
        <thead>
          <tr><th>#</th><th>Label</th><th>Conf.</th><th>Status</th><th></th></tr>
        </thead>
        <tbody>
          {cases.map((c) => (
            <React.Fragment key={c.id}>
              <tr className={c.label === "abnormal" ? "row-abn" : ""}>
                <td>{c.id}</td>
                <td>{c.label}</td>
                <td>{(c.probability * 100).toFixed(0)}%</td>
                <td><span className={`chip ${c.status}`}>{c.status}</span></td>
                <td>
                  <button className="link" onClick={() => setOpen(open === c.id ? null : c.id)}>
                    {open === c.id ? "hide" : "audit"}
                  </button>
                  {c.status === "pending" && (
                    <>
                      <button className="mini approve" onClick={() => onDecision(c.id, "approve")}>✓</button>
                      <button className="mini reject" onClick={() => onDecision(c.id, "reject")}>✗</button>
                    </>
                  )}
                </td>
              </tr>
              {open === c.id && (
                <tr><td colSpan={5}><AuditTrail caseId={c.id} /></td></tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}
