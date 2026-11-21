# Image for the scoring API and the Kafka consumer (no GNN/torch here - the GNN
# is trained offline and only its scores are consumed).
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; xgboost needs libgomp.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install the serving subset (skip heavy GNN/EDA libs for a lean image).
RUN pip install --no-cache-dir \
    "xgboost==3.4.1" "scikit-learn>=1.4" "pandas>=2.1" "numpy>=1.26" \
    "pyarrow>=15" "joblib>=1.3" "networkx>=3.2" \
    "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2.6" "kafka-python>=2.0"

COPY src ./src
COPY artifacts ./artifacts

EXPOSE 8000
# Model artifacts are expected under ./artifacts (bind-mount or baked in).
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
