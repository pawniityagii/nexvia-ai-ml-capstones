# Image for the Streamlit analyst dashboard.
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "streamlit>=1.33" "pandas>=2.1" "numpy>=1.26" "networkx>=3.2" \
    "xgboost==3.4.1" "scikit-learn>=1.4" "pyarrow>=15" "joblib>=1.3"

COPY src ./src
COPY dashboard ./dashboard
COPY artifacts ./artifacts

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
