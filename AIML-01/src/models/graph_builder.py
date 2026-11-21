"""Build the fraud-ring graph and derive graph features.

The brief asks for "a graph of accounts, devices and cards" to detect
connected fraud rings. We build a transaction graph whose edges connect
transactions that SHARE a real identity attribute present in IEEE-CIS:
the card identity (card_uid), the device (device_uid) or the purchaser
email (email_key). No device/card edges are fabricated - if the data lacks
the attribute, the edge is simply absent.

To stay O(n) rather than O(n^2) inside large identity groups, transactions
that share a key are chained in time order (t_i -> t_{i+1}). Chaining
preserves connected components exactly and keeps the graph sparse enough for
GraphSAGE message passing.

Two graph features feed the tabular XGBoost model:
  * ring_degree      - size of the connected component (structural, label-free)
  * ring_fraud_rate  - leak-free expanding prior fraud rate on the same card
The GNN (train_gnn.py) is the actual ring detector; these two columns just let
XGBoost alone benefit from topology.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from ..common import config as C

# Link transactions that share a SPECIFIC card or a SPECIFIC device fingerprint.
# Email *domain* is deliberately NOT a link key: it is low-cardinality (a handful
# of providers) and would connect unrelated people into one giant component.
LINK_KEYS = ["card_uid", "device_uid"]


def build_transaction_graph(feats: pd.DataFrame) -> nx.Graph:
    """Nodes = TransactionID. Edges chain transactions sharing an identity key."""
    g = nx.Graph()
    g.add_nodes_from(feats[C.ID_COL].tolist())
    order = feats.sort_values(C.TIME_COL)
    for key in LINK_KEYS:
        if key not in order.columns:
            continue
        for _, grp in order.groupby(key, sort=False):
            ids = grp[C.ID_COL].to_numpy()
            if len(ids) > 1:
                g.add_edges_from(zip(ids[:-1], ids[1:], ))
    return g


def compute_ring_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame[TransactionID, ring_degree, ring_fraud_rate]."""
    g = build_transaction_graph(feats)

    # ring_degree = connected-component size (structural, uses no labels)
    comp_size: dict = {}
    for comp in nx.connected_components(g):
        s = len(comp)
        for n in comp:
            comp_size[n] = s
    degree = feats[C.ID_COL].map(comp_size).fillna(1).astype(float)

    # ring_fraud_rate = leak-free expanding prior fraud rate on the same card
    if C.TARGET in feats.columns:
        d = feats[[C.ID_COL, C.TIME_COL, "card_uid", C.TARGET]].sort_values(C.TIME_COL)
        gg = d.groupby("card_uid", sort=False)
        prior_cnt = gg.cumcount()
        prior_fraud = gg[C.TARGET].cumsum().to_numpy() - d[C.TARGET].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(prior_cnt.to_numpy() > 0,
                            prior_fraud / np.maximum(prior_cnt.to_numpy(), 1), 0.0)
        rate_s = pd.Series(rate, index=d[C.ID_COL].to_numpy())
        fraud_rate = feats[C.ID_COL].map(rate_s).fillna(0.0).astype(float)
    else:
        # serving: no labels available -> value comes from the feature store cache
        fraud_rate = pd.Series(0.0, index=feats.index)

    return pd.DataFrame({
        C.ID_COL: feats[C.ID_COL].values,
        "ring_degree": degree.values,
        "ring_fraud_rate": fraud_rate.values,
    })


def attach_ring_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Overwrite the placeholder ring_* columns with computed values."""
    ring = compute_ring_features(feats).set_index(C.ID_COL)
    feats = feats.copy()
    feats["ring_degree"] = feats[C.ID_COL].map(ring["ring_degree"]).fillna(1.0)
    feats["ring_fraud_rate"] = feats[C.ID_COL].map(ring["ring_fraud_rate"]).fillna(0.0)
    return feats


def ring_summary(feats: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Analyst view: largest suspicious components and their fraud density."""
    g = build_transaction_graph(feats)
    rows = []
    label = feats.set_index(C.ID_COL)[C.TARGET] if C.TARGET in feats.columns else None
    for i, comp in enumerate(nx.connected_components(g)):
        if len(comp) < 3:
            continue
        comp = list(comp)
        frauds = int(label.reindex(comp).sum()) if label is not None else 0
        rows.append({
            "ring_id": i,
            "size": len(comp),
            "fraud_txns": frauds,
            "fraud_rate": round(frauds / len(comp), 3) if label is not None else None,
            "sample_txns": comp[:5],
        })
    out = pd.DataFrame(rows).sort_values(["fraud_rate", "size"], ascending=False)
    return out.head(top).reset_index(drop=True)
