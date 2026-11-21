# Individual Reflection - AIML-01

## What I built
The end-to-end real-time fraud detection pipeline: leak-free feature engineering
(velocity, amount-vs-average, device/region changes, graph ring features), an
XGBoost classifier handling the under-1% fraud rate with class weighting and a
precision-recall-tuned threshold, a fraud-ring graph that links transactions by
shared card and device, a FastAPI scoring service, and a Streamlit analyst
console with a case queue and feedback labels. A pytest suite, including a
leakage-guard test, backs the feature layer.

## What I learned
Most of the MLOps stack was new to me: Kafka streaming, the Feast feature store,
MLflow run tracking, Evidently drift monitoring, graph neural networks with
PyTorch Geometric, and Kubernetes manifests. The idea I understand best now is
why accuracy is the wrong metric for rare-event problems and how precision-recall
AUC and threshold tuning address it.

## Hardest problem
Keeping the time-based features leak-free - making sure every rolling statistic
for a transaction uses only earlier transactions - and getting the fraud-ring
graph to form meaningful connected components instead of one giant cluster (the
fix was linking by specific card and device identity, not by low-cardinality
fields like email domain).

## Design choices
The repository demonstrates the full pipeline on synthetically generated
transactions so it runs without a dataset download. The training and evaluation
code targets the IEEE-CIS dataset for reportable metrics, and the GraphSAGE model
trains on a GPU and blends its score with XGBoost.
