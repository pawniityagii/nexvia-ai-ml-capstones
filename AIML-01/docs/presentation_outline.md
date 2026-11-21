# 10-Minute Presentation Outline

Structure the final presentation (slides or screen recording) around this.
Keep one idea per slide; rehearse to ~1 minute each.

1. **Title (0:30)** — Real-Time Financial Fraud Detection Platform. Your name,
   Nexvia AIML-01. One line: "scores transactions in ms and finds fraud rings."

2. **Problem (1:00)** — Card fraud is rare (<1%) but costly; it must be caught
   in real time, and fraudsters operate in rings sharing cards/devices. Three
   hard parts: imbalance, latency, and linked accounts.

3. **Data (1:00)** — IEEE-CIS: ~590k transactions, ~3.5% fraud, real device/card
   fields. Show the class-imbalance bar and why accuracy is the wrong metric.

4. **Approach / architecture (1:30)** — Walk the architecture diagram:
   features → XGBoost + graph/GNN → blend → FastAPI scorer → dashboard, with
   MLflow, Feast, Kafka, Evidently, Docker/K8s around it.

5. **Feature engineering (1:00)** — Velocity, amount-vs-average, device/region
   change, graph ring features. Emphasise *leak-free* (and the test proving it).

6. **Models + results (1:30)** — XGBoost PR-AUC, GNN for rings, the blend lift.
   Show the real IEEE-CIS numbers and the PR curve. Explain threshold choice.

7. **Real-time demo (1:30)** — Screen-record: start the stream, watch flags
   appear, open a flagged case in the dashboard, confirm it, show a fraud ring.

8. **MLOps (1:00)** — MLflow runs, Feast train/serve parity, Evidently drift →
   retraining trigger, Docker/K8s deploy.

9. **Challenges + next steps (0:30)** — e.g. real-time GNN is hard (batch ring
   scoring + cached features was the pragmatic call); next: live graph updates,
   Redis online store, fairness audit across segments.

10. **Close (0:30)** — What you built and owned, and the one thing you learned.

**Tips:** lead with the demo video if live demo is risky. Put raw numbers on
slides, not in narration. Name the trade-offs — reviewers reward honesty about
what is approximate.
