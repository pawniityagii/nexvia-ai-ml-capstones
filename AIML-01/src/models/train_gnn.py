"""Graph neural network for fraud-ring detection (PyTorch Geometric).

Transactions are nodes; edges connect transactions that share a card, device
or email (see graph_builder). A 2-layer GraphSAGE learns from each node's
neighbourhood, so a transaction sitting inside a dense shared-infra cluster
(a ring) is pushed toward the fraud class even when its own features look
ordinary. Output: a per-transaction GNN fraud score, saved for blending with
XGBoost (see blend.py).

Trained on a GPU (e.g. Colab); runs on CPU for small graphs.
Requires: torch, torch_geometric.

Run:
    python -m src.models.train_gnn --epochs 60
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..common import config as C
from ..common import data_loader
from ..features import feature_engineering as fe
from . import graph_builder


def _build_pyg_data(feats: pd.DataFrame):
    import torch
    from torch_geometric.data import Data

    feats = feats.reset_index(drop=True)
    id_to_idx = {tid: i for i, tid in enumerate(feats[C.ID_COL].to_numpy())}

    # node features: standardised model matrix
    X = fe.to_model_matrix(feats).to_numpy(dtype="float32")
    mu, sigma = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sigma

    # edges from the shared-identity graph (undirected -> both directions)
    g = graph_builder.build_transaction_graph(feats)
    src, dst = [], []
    for a, b in g.edges():
        ia, ib = id_to_idx[a], id_to_idx[b]
        src += [ia, ib]
        dst += [ib, ia]
    if not src:                                   # degenerate: self-loops only
        src, dst = list(range(len(feats))), list(range(len(feats)))
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    y = torch.tensor(feats[C.TARGET].astype(int).to_numpy(), dtype=torch.long)
    data = Data(x=torch.tensor(X), edge_index=edge_index, y=y)

    # chronological masks
    order = np.argsort(feats[C.TIME_COL].to_numpy(), kind="stable")
    n = len(feats)
    tr, va = int(n * 0.7), int(n * 0.85)
    train_idx, val_idx, test_idx = order[:tr], order[tr:va], order[va:]
    for name, idx in [("train_mask", train_idx), ("val_mask", val_idx),
                      ("test_mask", test_idx)]:
        m = torch.zeros(n, dtype=torch.bool)
        m[idx] = True
        setattr(data, name, m)
    return data


def _model(in_dim: int, hidden: int = 64):
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    class GraphSAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
            self.lin = nn.Linear(hidden, 2)
            self.dropout = 0.3

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.lin(x)

    return GraphSAGE()


def train(epochs: int = 60, synthetic: bool = False) -> dict:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, roc_auc_score

    from ..common import tracking

    df = data_loader.load_raw(prefer_synthetic=synthetic)
    feats = fe.engineer(df)
    feats = graph_builder.attach_ring_features(feats)
    pre = fe.fit_encoders(feats)
    feats = fe.apply_encoders(feats, pre)

    data = _build_pyg_data(feats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = _model(data.x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    ytr = data.y[data.train_mask]
    w = torch.tensor([1.0, float((ytr == 0).sum() / max(1, (ytr == 1).sum()))],
                     device=device)

    with tracking.start_run(run_name="gnn-graphsage") as run:
        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask],
                                   weight=w)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            proba = F.softmax(model(data.x, data.edge_index), dim=1)[:, 1].cpu().numpy()
        yte = data.y[data.test_mask].cpu().numpy()
        pte = proba[data.test_mask.cpu().numpy()]
        metrics = {
            "gnn_test_pr_auc": float(average_precision_score(yte, pte)),
            "gnn_test_roc_auc": float(roc_auc_score(yte, pte)),
            "epochs": epochs, "nodes": int(data.num_nodes),
            "edges": int(data.edge_index.shape[1]),
        }
        run.log_metrics(metrics)
        torch.save(model.state_dict(), C.GNN_MODEL_PATH)
        pd.DataFrame({C.ID_COL: feats[C.ID_COL].to_numpy(),
                      "gnn_score": proba}).to_parquet(
            C.ARTIFACTS_DIR / "gnn_scores.parquet")
    print(metrics)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    train(a.epochs, a.synthetic)
