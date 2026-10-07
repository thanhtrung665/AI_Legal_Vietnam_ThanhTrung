import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import check_transformer_general

from churn.features.clean import clean_churn
from churn.features.transformers import FrequencyEncoder, Winsorizer


def test_winsorizer_clips_to_train_quantiles():
    X_train = np.arange(100, dtype=float).reshape(-1, 1)
    w = Winsorizer(0.05, 0.95).fit(X_train)
    out = w.transform(np.array([[-1000.0], [50.0], [1000.0]]))
    assert out[0, 0] == pytest.approx(np.quantile(X_train, 0.05))
    assert out[1, 0] == 50.0
    assert out[2, 0] == pytest.approx(np.quantile(X_train, 0.95))


def test_winsorizer_keeps_nan_and_feature_names():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 100.0]})
    w = Winsorizer(0.0, 0.5).set_output(transform="pandas").fit(X)
    out = w.transform(X)
    assert list(out.columns) == ["a"] and np.isnan(out.loc[1, "a"])


def test_winsorizer_passes_sklearn_transformer_check():
    check_transformer_general("Winsorizer", Winsorizer())


def test_frequency_encoder_unseen_is_zero():
    enc = FrequencyEncoder().fit(pd.DataFrame({"c": ["a", "a", "b", "c"]}))
    assert enc.transform(pd.DataFrame({"c": ["a", "z"]})).ravel().tolist() == [0.5, 0.0]


def test_clean_normalizes_and_dedups():
    raw = pd.DataFrame({
        "customer_id": ["C0000001", "C0000001"], "signup_date": ["2024-01-01", "2024-01-01"],
        "contract": [" Month-to-Month ", " Month-to-Month "], "payment_method": ["Cash", "Cash"],
        "region": ["R01", "R01"], "age": [150, 150], "monthly_charges": [50.0, 50.0],
    })
    out = clean_churn(raw)
    assert len(out) == 1
    assert out.loc[0, "contract"] == "month-to-month"
    assert np.isnan(out.loc[0, "age"])  # impossible age -> NaN, imputed later
