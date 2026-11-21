"""Synthetic transaction generator (IEEE-CIS-shaped).

Purpose: let the ENTIRE pipeline - features, training, graph, streaming,
dashboard - run end to end with zero Kaggle download, so the demo is
recordable and CI can run. This is clearly-labelled fake data used for
development and demo only. Real evaluation numbers come from the real
IEEE-CIS dataset (see data/README.md); nothing here is reported as a result.

Design choices that make it useful rather than noise:
  * Each "card" has a home device, email and billing region, and a
    card-specific spend distribution - so normal behaviour is learnable.
  * Fraud appears in two honest forms:
      1. lone fraud  - anomalous amount / new device / new region.
      2. fraud rings - clusters of cards that SHARE devices and emails and
         transact in short bursts. This is what gives the graph real
         connected-component structure for the GNN to detect.
Overall fraud rate is tuned to ~3.5% to mirror IEEE-CIS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_SEED

CARD_NETWORKS = ["visa", "mastercard", "american express", "discover"]
CARD_TYPES = ["debit", "credit"]
PRODUCTS = ["W", "C", "R", "H", "S"]
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
                 "aol.com", "icloud.com", "anonymous.com", "live.com"]
DEVICE_TYPES = ["desktop", "mobile"]
OSES = ["Windows 10", "iOS 14.2", "Android 10", "Mac OS X", "Windows 7"]
BROWSERS = ["chrome 86.0", "safari 14.0", "mobile safari 14.0",
            "firefox 82.0", "edge 18.0"]
DEVICE_MODELS = ["Windows", "iOS Device", "SM-G973F", "MacBookPro", "rv:82.0",
                 "Pixel 4", "SM-A505F", "Trident/7.0"]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate(
    n_transactions: int = 60_000,
    n_cards: int = 6_000,
    n_rings: int = 40,
    target_fraud_rate: float = 0.035,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return a DataFrame with IEEE-CIS-style columns and an ``isFraud`` label."""
    rng = _rng(seed)

    # ---- build the card population -------------------------------------- #
    card1 = rng.integers(1000, 18000, size=n_cards)
    card2 = rng.integers(100, 600, size=n_cards)
    card3 = rng.choice([150, 185], size=n_cards)
    card5 = rng.integers(100, 240, size=n_cards)
    card4 = rng.choice(CARD_NETWORKS, size=n_cards, p=[0.55, 0.32, 0.08, 0.05])
    card6 = rng.choice(CARD_TYPES, size=n_cards, p=[0.6, 0.4])
    addr1 = rng.integers(100, 540, size=n_cards)
    addr2 = rng.choice([87, 60, 96], size=n_cards)
    home_email = rng.choice(EMAIL_DOMAINS, size=n_cards)
    home_device_idx = rng.integers(0, len(DEVICE_MODELS), size=n_cards)
    home_os = rng.choice(OSES, size=n_cards)
    home_browser = rng.choice(BROWSERS, size=n_cards)
    home_dtype = rng.choice(DEVICE_TYPES, size=n_cards, p=[0.45, 0.55])
    card_mean_amt = rng.gamma(shape=2.0, scale=45.0, size=n_cards) + 10
    # Each card has a UNIQUE device fingerprint (model + per-card serial), so
    # legitimate cards never share a device_uid. Fraud rings override this with
    # one shared fingerprint, which is what creates ring structure in the graph.
    card_device_fp = np.array(
        [f"{DEVICE_MODELS[i]}#{c:05d}" for c, i in enumerate(home_device_idx)])

    # cards that belong to a fraud ring, and the shared infra of each ring
    ring_of_card = np.full(n_cards, -1, dtype=int)
    ring_members: dict[int, list[int]] = {}
    ring_pool = rng.choice(n_cards, size=min(n_rings * 20, n_cards), replace=False)
    for r in range(n_rings):
        members = ring_pool[r * 20:(r + 1) * 20]
        if len(members) == 0:
            continue
        ring_of_card[members] = r
        ring_members[r] = members.tolist()
    # A ring shares its ENTIRE device fingerprint (the DEVICE_KEYS tuple), so all
    # members hash to one device_uid and form a single connected component.
    ring_device_fp = {r: f"{DEVICE_MODELS[rng.integers(0, len(DEVICE_MODELS))]}#R{r:03d}"
                      for r in ring_members}
    ring_email = {r: rng.choice(EMAIL_DOMAINS) for r in ring_members}
    ring_browser = {r: rng.choice(BROWSERS) for r in ring_members}
    ring_os = {r: rng.choice(OSES) for r in ring_members}
    ring_dtype = {r: rng.choice(DEVICE_TYPES) for r in ring_members}

    rows = []
    n_fraud_target = int(n_transactions * target_fraud_rate)
    fraud_budget = n_fraud_target

    # ---- ring fraud first (bursty, shared infra) ------------------------ #
    ring_fraud = int(n_fraud_target * 0.45)
    per_ring = max(1, ring_fraud // max(1, len(ring_members)))
    for r, members in ring_members.items():
        base_t = rng.integers(86_400, 15_000_000)
        for _ in range(per_ring):
            if fraud_budget <= 0:
                break
            c = int(rng.choice(members))
            t = int(base_t + rng.integers(0, 7_200))        # short burst window
            amt = float(rng.gamma(2.2, 90) + 30)            # elevated amounts
            rows.append(_row(
                rng, c, t, amt, is_fraud=1,
                card1=card1, card2=card2, card3=card3, card5=card5,
                card4=card4, card6=card6, addr1=addr1, addr2=addr2,
                email=ring_email[r],
                device_fp=ring_device_fp[r], os_=ring_os[r], browser=ring_browser[r],
                dtype=ring_dtype[r], card_mean=card_mean_amt[c]))
            fraud_budget -= 1

    # ---- lone fraud (anomalous device / region / amount) ---------------- #
    while fraud_budget > 0:
        c = int(rng.integers(0, n_cards))
        t = int(rng.integers(86_400, 15_000_000))
        amt = float(rng.gamma(2.5, 120) + 50)
        rows.append(_row(
            rng, c, t, amt, is_fraud=1,
            card1=card1, card2=card2, card3=card3, card5=card5,
            card4=card4, card6=card6, addr1=addr1, addr2=addr2,
            email=rng.choice(EMAIL_DOMAINS),                # new email
            device_fp=f"{rng.choice(DEVICE_MODELS)}#X{rng.integers(0, 10**9)}",  # new device
            os_=rng.choice(OSES), browser=rng.choice(BROWSERS),
            dtype=rng.choice(DEVICE_TYPES), card_mean=card_mean_amt[c]))
        fraud_budget -= 1

    # ---- legitimate transactions --------------------------------------- #
    n_legit = n_transactions - len(rows)
    for _ in range(max(0, n_legit)):
        c = int(rng.integers(0, n_cards))
        t = int(rng.integers(86_400, 15_000_000))
        amt = float(max(1.0, rng.normal(card_mean_amt[c], card_mean_amt[c] * 0.3)))
        rows.append(_row(
            rng, c, t, amt, is_fraud=0,
            card1=card1, card2=card2, card3=card3, card5=card5,
            card4=card4, card6=card6, addr1=addr1, addr2=addr2,
            email=home_email[c],
            device_fp=card_device_fp[c], os_=home_os[c],
            browser=home_browser[c], dtype=home_dtype[c],
            card_mean=card_mean_amt[c]))

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "TransactionID", np.arange(2_987_000, 2_987_000 + len(df)))
    return df


def _row(rng, c, t, amt, is_fraud, *, card1, card2, card3, card5, card4, card6,
         addr1, addr2, email, device_fp, os_, browser, dtype, card_mean) -> dict:
    """Assemble one IEEE-CIS-shaped record for card index ``c``."""
    # C/D counting+timedelta columns: cheap correlates, noisier for fraud.
    noise = 1.5 if is_fraud else 1.0
    return {
        "TransactionDT": int(t),
        "TransactionAmt": round(amt, 2),
        "ProductCD": rng.choice(PRODUCTS, p=[0.74, 0.11, 0.07, 0.05, 0.03]),
        "card1": int(card1[c]), "card2": int(card2[c]), "card3": int(card3[c]),
        "card5": int(card5[c]), "card4": card4[c], "card6": card6[c],
        "addr1": int(addr1[c]), "addr2": int(addr2[c]),
        "dist1": float(abs(rng.normal(10, 30)) * noise),
        "P_emaildomain": email,
        "R_emaildomain": rng.choice(EMAIL_DOMAINS) if rng.random() < 0.3 else email,
        "DeviceType": dtype,
        "DeviceInfo": device_fp,
        "id_30": os_, "id_31": browser,
        "C1": int(abs(rng.normal(1, 2)) * noise), "C2": int(abs(rng.normal(1, 2)) * noise),
        "C13": int(abs(rng.normal(1, 5)) * noise), "C14": int(abs(rng.normal(1, 4)) * noise),
        "D1": float(abs(rng.normal(50, 60))), "D4": float(abs(rng.normal(30, 50))),
        "D10": float(abs(rng.normal(40, 55))), "D15": float(abs(rng.normal(35, 50))),
        "isFraud": int(is_fraud),
    }


if __name__ == "__main__":
    import argparse
    from .config import SYNTHETIC_CSV

    p = argparse.ArgumentParser(description="Generate synthetic fraud data.")
    p.add_argument("--n", type=int, default=60_000)
    p.add_argument("--cards", type=int, default=6_000)
    p.add_argument("--rings", type=int, default=40)
    p.add_argument("--out", type=str, default=str(SYNTHETIC_CSV))
    args = p.parse_args()

    out = generate(args.n, args.cards, args.rings)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out):,} rows ({out['isFraud'].mean():.3%} fraud) -> {args.out}")
