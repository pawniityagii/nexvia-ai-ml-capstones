"""Feast feature definitions.

Feast is the brief's answer to train/serve skew: the SAME card-level features
are written once and read by both training (offline) and the scoring service
(online). The entity is the card (card_uid). The source parquet is produced by
scripts/export_feast_source.py.

Pinned against the Feast API in requirements.txt. Apply with:
    cd feature_repo && feast apply && feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

card = Entity(name="card", join_keys=["card_uid"], description="Card identity")

card_source = FileSource(
    name="card_features_source",
    path="data/feast_card_features.parquet",
    timestamp_field="event_timestamp",
)

card_features = FeatureView(
    name="card_features",
    entities=[card],
    ttl=timedelta(days=90),
    schema=[
        Field(name="txn_count", dtype=Int64),
        Field(name="amt_mean", dtype=Float32),
        Field(name="amt_std", dtype=Float32),
        Field(name="ring_degree", dtype=Float32),
        Field(name="ring_fraud_rate", dtype=Float32),
    ],
    source=card_source,
    online=True,
)
