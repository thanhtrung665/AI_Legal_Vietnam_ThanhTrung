"""Behavioural tests (Ribeiro et al., 2020, 'CheckList'): invariance, directional, robustness."""

import pandas as pd
import pytest

BASE = pd.DataFrame([{
    "customer_id": "C0000001", "signup_date": pd.Timestamp("2024-01-01"), "age": 35.0,
    "tenure_months": 24, "monthly_charges": 70.0, "total_charges": 1680.0, "contract": "one-year",
    "payment_method": "credit-card", "region": "r01", "support_calls": 1, "data_usage_gb": 12.0,
}])


@pytest.fixture
def score(trained_model):
    return lambda **kw: float(trained_model.predict_proba(BASE.assign(**kw))[:, 1][0])


def test_invariance_to_customer_id(score):
    assert score(customer_id="C9999999") == pytest.approx(score())


def test_directional_support_calls(score):
    assert score(support_calls=8) > score(support_calls=0)


def test_directional_contract(score):
    assert score(contract="month-to-month") > score(contract="two-year")


def test_robust_to_missing_and_unseen(score):
    assert 0.0 <= score(age=None, payment_method="crypto", region="r99") <= 1.0
