"""Kafka producer that replays transactions as a live stream.

Reads the dataset row by row and publishes each transaction as JSON to the
``transactions`` topic, simulating a payment feed. Rate is configurable.

Run (needs a broker; docker-compose up kafka):
    python -m src.streaming.producer --rate 50
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from ..common import config as C
from ..common import data_loader


def main(rate: float, limit: int | None):
    from kafka import KafkaProducer

    df = data_loader.load_raw()
    if limit:
        df = df.head(limit)
    producer = KafkaProducer(
        bootstrap_servers=C.KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
    )
    delay = 1.0 / rate if rate > 0 else 0.0
    sent = 0
    for _, row in df.iterrows():
        rec = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        producer.send(C.KAFKA_TOPIC, key=rec.get(C.ID_COL), value=rec)
        sent += 1
        if sent % 500 == 0:
            producer.flush()
            print(f"sent {sent:,}")
        if delay:
            time.sleep(delay)
    producer.flush()
    print(f"done: {sent:,} transactions -> topic '{C.KAFKA_TOPIC}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rate", type=float, default=50, help="transactions per second")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    main(a.rate, a.limit)
