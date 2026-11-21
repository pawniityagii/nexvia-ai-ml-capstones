import React, { useRef } from "react";

export default function UploadPanel({ onAnalyze, busy }) {
  const ref = useRef();
  return (
    <div className="panel">
      <h2>Upload scan</h2>
      <p className="hint">PNG, JPG or DICOM (.dcm)</p>
      <input ref={ref} type="file" accept=".png,.jpg,.jpeg,.dcm,.dicom" />
      <button
        disabled={busy}
        onClick={() => {
          const f = ref.current?.files?.[0];
          if (f) onAnalyze(f);
        }}
      >
        {busy ? "Analyzing..." : "Analyze"}
      </button>
    </div>
  );
}
