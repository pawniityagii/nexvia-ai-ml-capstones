"""Train the DenseNet121 chest-X-ray classifier.

Requires torch + torchvision (Colab/GPU). Reads data/labels.csv (produced by the
synthetic generator or built from a real dataset), does a stratified split,
transfer-learns DenseNet121, evaluates with ROC-AUC + sensitivity/specificity
(the metrics that matter for screening), and saves the checkpoint + metrics.
MLflow is used when available; otherwise metrics are written to artifacts/.

Run (after data exists):
    python -m src.training.train_classifier --epochs 10
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from ..common import config as C
from ..common import preprocessing as pp


def _read_labels():
    rows = []
    with open(C.LABELS_CSV) as f:
        for r in csv.DictReader(f):
            rows.append((r["filename"], 1 if r["label"] == "abnormal" else 0))
    return rows


def _make_dataset(rows, train: bool):
    import torch
    from torch.utils.data import Dataset

    class XRayDataset(Dataset):
        def __init__(self, items, train):
            self.items = items
            self.train = train
            self.rng = np.random.default_rng(C.RANDOM_SEED)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            fname, label = self.items[i]
            gray = pp.load_image(C.IMAGES_DIR / fname)
            if self.train:
                gray = pp.augment(gray, self.rng)
            x = pp.to_model_input(gray)
            return torch.from_numpy(x).float(), int(label)

    return XRayDataset(rows, train)


def train(epochs: int = 10, arch: str = "densenet121", batch_size: int = 32,
          lr: float = 1e-4) -> dict:
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from ..models.classifier import build_model

    rows = _read_labels()
    rng = np.random.default_rng(C.RANDOM_SEED)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * 0.2))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, pretrained=True).to(device)
    train_dl = DataLoader(_make_dataset(train_rows, True), batch_size=batch_size,
                          shuffle=True, num_workers=2)
    val_dl = DataLoader(_make_dataset(val_rows, False), batch_size=batch_size)

    # class weighting for imbalance
    pos = sum(l for _, l in train_rows)
    neg = len(train_rows) - pos
    weight = torch.tensor([1.0, neg / max(1, pos)], device=device)
    crit = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    try:
        import mlflow
        mlflow.set_experiment("chestxray-classifier")
        run_ctx = mlflow.start_run(run_name=arch)
    except Exception:
        run_ctx = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

    # evaluation
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in val_dl:
            p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
            probs += p.tolist()
            ys += y.tolist()
    probs, ys = np.array(probs), np.array(ys)
    preds = (probs >= C.DEFAULT_THRESHOLD).astype(int)
    tp = int(((preds == 1) & (ys == 1)).sum()); fn = int(((preds == 0) & (ys == 1)).sum())
    tn = int(((preds == 0) & (ys == 0)).sum()); fp = int(((preds == 1) & (ys == 0)).sum())
    metrics = {
        "arch": arch, "n_train": len(train_rows), "n_val": len(val_rows),
        "val_auc": float(roc_auc_score(ys, probs)) if len(set(ys)) > 1 else None,
        "accuracy": float((preds == ys).mean()),
        "sensitivity": tp / (tp + fn) if (tp + fn) else None,
        "specificity": tn / (tn + fp) if (tn + fp) else None,
    }

    torch.save({"model": model.state_dict(), "arch": arch}, C.CLASSIFIER_PATH)
    C.CLASSIFIER_META.write_text(json.dumps({"arch": arch, "classes": C.CLASSES}, indent=2))
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    if run_ctx is not None:
        import mlflow
        mlflow.log_metrics({k: v for k, v in metrics.items()
                            if isinstance(v, (int, float))})
        mlflow.end_run()
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--arch", default="densenet121",
                   choices=["densenet121", "efficientnet_b0"])
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    train(a.epochs, a.arch, a.batch_size)
