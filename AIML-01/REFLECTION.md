# Individual Reflection — AIML-01

> Draft to make your own. Edit it into your voice and keep only what is true for
> you before submitting; a reviewer may ask about anything written here.

## What I built and own
I built the end-to-end real-time fraud detection pipeline: leak-free feature
engineering (velocity, amount-vs-average, device/region changes, graph ring
features), an XGBoost classifier handling the under-1% fraud rate with class
weighting and a PR-curve-tuned threshold, a fraud-ring graph that links
transactions by shared card and device, a FastAPI scoring service, and a
Streamlit analyst console with a case queue and feedback labels. A pytest suite,
including a leakage-guard test, backs the feature layer.

## What I learned
Most of the MLOps stack was new to me: Kafka streaming, the Feast feature store,
MLflow run tracking, Evidently drift monitoring, graph neural networks with
PyTorch Geometric, and Kubernetes manifests. The idea I understand best now is
why accuracy is the wrong metric for rare-event problems and how PR-AUC and
threshold tuning address it.

## Hardest problem
Keeping the time-based features leak-free — making sure every rolling statistic
for a transaction uses only earlier transactions — and getting the fraud-ring
graph to form meaningful connected components instead of one giant blob (the fix
was linking by specific card/device identity, not by low-cardinality fields like
email domain).

## What is real vs next
The pipeline runs end to end on synthetic data, which is separable by
construction, so those numbers are not results. The next step I would take is
training on the real IEEE-CIS dataset for reportable PR-AUC, and training the GNN
on a GPU to blend its score with XGBoost.
