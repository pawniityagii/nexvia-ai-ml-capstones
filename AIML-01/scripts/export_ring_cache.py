"""Write artifacts/ring_cache.json: card_uid -> {ring_degree, ring_fraud_rate}.

The online scorer looks up these precomputed graph features by card at serving
time (the production equivalent is the Feast online store). Run after training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config as C
from src.common import data_loader
from src.features import feature_engineering as fe
from src.models import graph_builder


def main():
    df = data_loader.load_raw()
    feats = fe.engineer(df)
    feats = graph_builder.attach_ring_features(feats)
    # last-seen ring feature per card is what the online store would serve
    cache = (feats.sort_values(C.TIME_COL)
             .groupby("card_uid")
             .agg(ring_degree=("ring_degree", "max"),
                  ring_fraud_rate=("ring_fraud_rate", "last"))
             .to_dict("index"))
    (C.ARTIFACTS_DIR / "ring_cache.json").write_text(json.dumps(cache))
    print(f"wrote ring cache for {len(cache):,} cards")


if __name__ == "__main__":
    main()
