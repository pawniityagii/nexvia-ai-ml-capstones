"""Kafka consumer that scores each transaction in real time.

Subscribes to the ``transactions`` topic, scores every message with the shared
Scorer (same code path as the HTTP API), and persists flagged cases to the case
store for the analyst dashboard.

Run (needs a broker + a trained model):
    python -m src.streaming.consumer
"""
from __future__ import annotations

import argparse
import json

from ..common import config as C
from ..serving.scorer import Scorer


def main(from_beginning: bool):
    from kafka import KafkaConsumer

    scorer = Scorer(write_cases=True)
    consumer = KafkaConsumer(
        C.KAFKA_TOPIC,
        bootstrap_servers=C.KAFKA_BOOTSTRAP,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest" if from_beginning else "latest",
        enable_auto_commit=True,
        group_id="fraud-scorer",
    )
    print(f"listening on '{C.KAFKA_TOPIC}' ...")
    n = flagged = 0
    for msg in consumer:
        res = scorer.score(msg.value)
        n += 1
        if res["decision"] == "flag":
            flagged += 1
            print(f"[FLAG] txn={res['transaction_id']} "
                  f"p={res['fraud_probability']:.3f} {res['latency_ms']}ms")
        if n % 1000 == 0:
            print(f"scored {n:,} ({flagged:,} flagged)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--from-beginning", action="store_true")
    a = p.parse_args()
    main(a.from_beginning)
