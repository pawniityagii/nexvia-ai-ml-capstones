# Datasets

Datasets are **not committed** (see `.gitignore`). Two options:

## Option A — Synthetic (default, zero download)
Nothing to do. If no real CSV is present, the pipeline generates
`synthetic_transactions.csv` automatically. Regenerate explicitly with:
```bash
python -m src.common.synthetic --n 60000 --cards 6000 --rings 40
```
Synthetic data is separable by construction — use it to prove the pipeline
runs, not to report accuracy.

## Option B — IEEE-CIS Fraud Detection (real evaluation)
Primary dataset for this project (chosen because it has real device, card, and
address fields, so the fraud-ring graph in step 4 is honest).

1. Kaggle → *IEEE-CIS Fraud Detection* competition → Data.
   <https://www.kaggle.com/competitions/ieee-fraud-detection/data>
   (Accept the competition rules; download requires a Kaggle account.)
2. Place these files here in `data/`:
   - `train_transaction.csv`
   - `train_identity.csv`  (optional but recommended — adds device/OS/browser)
3. The loader merges them on `TransactionID` automatically.

Via the Kaggle CLI:
```bash
pip install kaggle          # put kaggle.json in ~/.kaggle/
kaggle competitions download -c ieee-fraud-detection -p data/
unzip data/ieee-fraud-detection.zip -d data/
```

### Other options named in the brief
PaySim, Credit Card Fraud (ULB), Bank Account Fraud (BAF), and the Elliptic
Bitcoin graph are all valid. The code keys off IEEE-CIS column names; using a
different dataset means adjusting the column groups in `src/common/config.py`.
Elliptic is the easiest *ready-made* graph for the GNN but does not blend with a
tabular transaction model.
