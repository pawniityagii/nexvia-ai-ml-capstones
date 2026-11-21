# Backend image (FastAPI + fallback predictor). The deep-learning stack is NOT
# installed here - the trained model is mounted via ./artifacts. Without it the
# service runs the deterministic fallback, so the container always starts.
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2.6" \
    "python-multipart>=0.0.9" "pillow>=10.0" "numpy>=1.26" \
    "scikit-image>=0.22" "pydicom>=2.4" "requests>=2.31" "scikit-learn>=1.4"

COPY src ./src
COPY artifacts ./artifacts

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
