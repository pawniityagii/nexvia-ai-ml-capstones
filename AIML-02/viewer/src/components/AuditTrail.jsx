import React, { useEffect, useState } from "react";
import { getAudit } from "../api.js";

export default function AuditTrail({ caseId }) {
  const [rows, setRows] = useState([]);
  useEffect(() => {
    getAudit(caseId).then(setRows).catch(() => setRows([]));
  }, [caseId]);
  if (!rows.length) return null;
  return (
    <div className="audit">
      <strong>Audit trail</strong>
      <ul>
        {rows.map((r) => (
          <li key={r.id}>
            <span className="a-action">{r.action}</span> by {r.actor}
            {r.note ? ` - ${r.note}` : ""}
            <span className="a-ts">{new Date(r.ts * 1000).toLocaleString()}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
