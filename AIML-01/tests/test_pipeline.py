"""Tests for the feature + graph layer.

The headline test is the leakage guard: a card's first-ever transaction must
carry no-history feature values. If future information ever leaks into a
feature, these assertions break.
"""
import numpy as np
import pandas as pd
import pytest

from src.common import synthetic
from src.features import feature_engineering as fe
from src.models import graph_builder


@pytest.fixture(scope="module")
def feats():
    df = synthetic.generate(n_transactions=4000, n_cards=400, n_rings=8, seed=7)
    f = fe.engineer(df)
    return graph_builder.attach_ring_features(f)


def test_no_nan_or_inf(feats):
    X = fe.to_model_matrix(feats)
    arr = X.to_numpy()
    assert np.isfinite(arr).all(), "model matrix has NaN/inf"


def test_feature_count_matches_config(feats):
    from src.common import config as C
    X = fe.to_model_matrix(feats)
    assert list(X.columns) == C.MODEL_FEATURES


def test_leakage_guard_first_txn_has_no_history():
    """First transaction on each card must use no-history defaults."""
    df = synthetic.generate(n_transactions=3000, n_cards=300, n_rings=5, seed=11)
    f = fe.engineer(df)
    f = f.merge(df[["TransactionID"]], on="TransactionID", how="left")
    order = f.sort_values("TransactionDT")
    first = order.groupby("card_uid", sort=False).head(1)
    # a card's first txn: zero velocity, default recency, ratio 1.0, zero z-score
    assert (first["txn_velocity_card_1h"] == 0).all()
    assert (first["txn_velocity_card_24h"] == 0).all()
    assert (first["secs_since_prev_card_txn"] >= 1e6).all()
    assert np.allclose(first["amt_to_card_mean"], 1.0)
    assert np.allclose(first["amt_to_card_std"], 0.0)
    assert (first["device_change_flag"] == 0).all()
    assert (first["addr_change_flag"] == 0).all()


def test_ring_features_present(feats):
    assert feats["ring_degree"].min() >= 1
    assert feats["ring_fraud_rate"].between(0, 1).all()


def test_unseen_category_is_safe():
    df = synthetic.generate(n_transactions=1000, n_cards=100, n_rings=2, seed=1)
    train = fe.engineer(df)
    pre = fe.fit_encoders(train)
    novel = df.head(5).copy()
    novel["P_emaildomain"] = "brand-new-domain-not-in-train.xyz"
    out = fe.apply_encoders(fe.engineer(novel), pre)
    assert out["P_emaildomain__enc"].notna().all()


def test_graph_components_separate_fraud(feats):
    """Fraud transactions should sit in larger components on average."""
    by_class = feats.groupby("isFraud")["ring_degree"].mean()
    assert by_class.loc[1] > by_class.loc[0]
