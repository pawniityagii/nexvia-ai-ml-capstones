# Model Card - Fraud Detection Platform

**For education only. Not a production or regulated fraud-decisioning system.**
A score here is a model estimate of risk, not a determination that a person or
transaction is fraudulent.

## Intended use
- Internship / portfolio demonstration of a real-time fraud ML pipeline.
- Showing an analyst-in-the-loop review workflow.

## Out of scope / do not use for
- Real financial decisions, account blocking, or anything affecting a real
  person. Any real deployment would need fairness review, regulatory sign-off,
  human adjudication, and auditing far beyond this project.

## Models
- **XGBoost** gradient-boosted trees on engineered tabular features.
- **GraphSAGE GNN** (PyTorch Geometric) on the card/device transaction graph.
- **Blend:** weighted average of the two scores; weight tuned on validation.

## Data
- **IEEE-CIS Fraud Detection** (Kaggle) for real evaluation, or a synthetic
  generator for development/demo. Synthetic metrics are near-perfect *by
  construction* and are never reported as results.

## Evaluation
- Primary metric **precision-recall AUC** (PR-AUC), because accuracy is
  misleading at <1% fraud. ROC-AUC and F1 at the tuned threshold are also
  reported. Splits are **chronological** (no future leakage).

## Limitations
- Trained on historical, possibly non-representative data; fraud patterns drift.
- Frequency-encoded categoricals lose information vs the raw identity.
- The online feature state is in-memory (not persisted/HA).
- Real-time GNN inference is approximated by cached ring features, not a live
  graph update per transaction.

## Bias & fairness considerations
- Email domain, device and billing-region signals can correlate with geography
  or socioeconomic factors. Flag rates should be checked across segments before
  any real use. No protected-attribute data is used, but proxies may exist.
- A false positive blocks a legitimate customer; a false negative lets fraud
  through. The decision threshold encodes that trade-off and should be set with
  business + compliance input, not left at a default.

## Human oversight
Every flagged transaction becomes an analyst case with confirm/clear actions and
a feedback label. The model supports analysts; it does not act autonomously.
