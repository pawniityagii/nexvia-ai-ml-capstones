"""Central configuration: paths, column groups, and the canonical feature list.

The feature list here is the single source of truth shared by the training
pipeline and the online scoring service. Keeping it in one place is what makes
train/serve parity honest - Feast reads the same list (see feature_repo/).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]          # repo root (aiml01/)
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"                  # trained models, encoders, metrics
MLRUNS_DIR = ROOT / "mlruns"                         # local MLflow backend store

RAW_TRANSACTION_CSV = DATA_DIR / "train_transaction.csv"   # IEEE-CIS
RAW_IDENTITY_CSV = DATA_DIR / "train_identity.csv"         # IEEE-CIS
SYNTHETIC_CSV = DATA_DIR / "synthetic_transactions.csv"    # generated, no download

XGB_MODEL_PATH = ARTIFACTS_DIR / "xgb_model.json"
GNN_MODEL_PATH = ARTIFACTS_DIR / "gnn_model.pt"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
FEATURE_META_PATH = ARTIFACTS_DIR / "feature_meta.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
REFERENCE_SAMPLE_PATH = ARTIFACTS_DIR / "reference_sample.parquet"  # for drift baseline

for _d in (DATA_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset schema (IEEE-CIS subset we actually use)
# --------------------------------------------------------------------------- #
TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"          # seconds offset from a reference point
AMOUNT_COL = "TransactionAmt"

# Raw categorical columns kept from IEEE-CIS (identity-linking attributes).
CATEGORICAL_RAW = [
    "ProductCD",
    "card4",            # card network (visa / mastercard / ...)
    "card6",            # card type (debit / credit)
    "P_emaildomain",    # purchaser email domain
    "R_emaildomain",    # recipient email domain
    "DeviceType",       # desktop / mobile
    "DeviceInfo",       # device string
    "id_30",            # OS
    "id_31",            # browser
]

# Raw numeric columns kept from IEEE-CIS.
NUMERIC_RAW = [
    AMOUNT_COL,
    "card1", "card2", "card3", "card5",   # card identifiers (numeric)
    "addr1", "addr2",                     # billing region
    "dist1",                              # distance
    "C1", "C2", "C13", "C14",             # counting features
    "D1", "D4", "D10", "D15",             # timedelta features
]

# The attributes we hash into node identities for the fraud-ring graph.
# These are the "accounts, devices and cards" the brief asks for and that
# IEEE-CIS actually contains - we do NOT fabricate device/card edges.
CARD_KEYS = ["card1", "card2", "card3", "card5", "card4", "card6", "addr1", "addr2"]
DEVICE_KEYS = ["DeviceInfo", "id_30", "id_31", "DeviceType"]
EMAIL_KEYS = ["P_emaildomain"]

# Engineered feature names (produced by features/feature_engineering.py).
ENGINEERED_NUMERIC = [
    "amt_log",                 # log1p(TransactionAmt)
    "amt_to_card_mean",        # amount vs this card's historical average
    "amt_to_card_std",         # z-ish score vs card history
    "txn_velocity_card_1h",    # #txns on this card in the trailing hour
    "txn_velocity_card_24h",   # #txns on this card in the trailing day
    "secs_since_prev_card_txn",# recency on this card
    "device_change_flag",      # device differs from card's last seen device
    "addr_change_flag",        # billing region differs from card's last seen
    "hour_of_day",
    "ring_degree",             # graph: #transactions sharing this card's ring
    "ring_fraud_rate",         # graph: historical fraud rate within the ring
]

# The full ordered feature vector the model consumes.
MODEL_FEATURES = NUMERIC_RAW + ENGINEERED_NUMERIC + [f"{c}__enc" for c in CATEGORICAL_RAW]

RANDOM_SEED = 42

# MLflow
MLFLOW_EXPERIMENT = "fraud-detection"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"file://{MLRUNS_DIR}")

# Kafka
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "transactions")

# Decision threshold (tuned on PR curve during training; overwritten in metrics.json).
DEFAULT_THRESHOLD = 0.5
