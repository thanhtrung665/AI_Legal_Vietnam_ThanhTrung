"""Custom scikit-learn transformers that follow the estimator API contract.

Rules from the scikit-learn developer guide ("Developing scikit-learn estimators"):
* ``__init__`` only stores parameters, unchanged and with no validation logic;
* everything learned in ``fit`` ends with a trailing underscore (``lower_``);
* ``fit`` returns ``self``; ``transform`` checks the estimator is fitted;
* ``get_feature_names_out`` makes the transformer work with ``set_output(transform="pandas")``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Winsorizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each column to quantiles learned on the training data. NaN is preserved."""

    def __init__(self, q_low: float = 0.01, q_high: float = 0.99):
        self.q_low = q_low
        self.q_high = q_high

    def fit(self, X, y=None):
        if not 0 <= self.q_low < self.q_high <= 1:
            raise ValueError("Require 0 <= q_low < q_high <= 1")
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        self.lower_ = np.nanquantile(arr, self.q_low, axis=0)
        self.upper_ = np.nanquantile(arr, self.q_high, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_", "upper_"])
        arr = np.asarray(X, dtype=float)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {arr.shape[1]}")
        return np.clip(arr, self.lower_, self.upper_)


class FrequencyEncoder(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Replace each category with its frequency in the training data (unseen -> 0)."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.maps_ = [X[c].value_counts(normalize=self.normalize).to_dict() for c in X.columns]
        return self

    def transform(self, X):
        check_is_fitted(self, "maps_")
        X = pd.DataFrame(X)
        return np.column_stack(
            [X.iloc[:, i].map(m).astype(float).fillna(0.0).to_numpy() for i, m in enumerate(self.maps_)]
        )


class ChurnFeatures(TransformerMixin, BaseEstimator):
    """Stateless domain features computed row by row.

    Deliberately no "days since signup relative to a fixed date": a reference date learned on
    the training period gives negative/shifted values on future data (see handbook ch. 5.6).
    Tenure already measures customer age relative to each customer's own scoring time.
    """

    DROP = ("customer_id", "signup_date")

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.feature_names_out_ = np.asarray(self.transform(X.head(2)).columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        signup = pd.to_datetime(X["signup_date"])
        tenure = X["tenure_months"].clip(lower=1)
        X["signup_month"] = signup.dt.month
        X["signup_dow"] = signup.dt.dayofweek
        X["avg_charge_per_month"] = X["total_charges"] / tenure
        X["charge_vs_expected"] = X["total_charges"] / (X["monthly_charges"] * tenure).clip(lower=1)
        X["calls_per_year"] = X["support_calls"] / (tenure / 12)
        X["is_new_customer"] = (X["tenure_months"] <= 3).astype(int)
        X["high_value"] = (X["monthly_charges"] > 100).astype(int)
        return X.drop(columns=[c for c in self.DROP if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_
