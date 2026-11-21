import React from "react";

const DEFAULT =
  "FOR EDUCATION ONLY - NOT FOR CLINICAL USE. Outputs are model estimates and " +
  "must be reviewed by a qualified radiologist.";

export default function DisclaimerBanner({ text }) {
  return <div className="disclaimer">{text || DEFAULT}</div>;
}
