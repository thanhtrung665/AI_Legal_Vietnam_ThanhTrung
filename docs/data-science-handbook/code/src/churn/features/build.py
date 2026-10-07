"""Leakage-safe preprocessing pipeline: every stateful step is fitted on training folds only."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from churn.features.transformers import ChurnFeatures, Winsorizer

NUM_COLS = [
    "age", "tenure_months", "monthly_charges", "total_charges", "support_calls", "data_usage_gb",
    "avg_charge_per_month", "charge_vs_expected", "calls_per_year",
]
LOW_CARD_COLS = ["contract", "payment_method"]
HIGH_CARD_COLS = ["region"]
PASSTHROUGH = ["is_new_customer", "high_value", "signup_month", "signup_dow"]
RAW_FEATURES = [
    "customer_id", "signup_date", "age", "tenure_months", "monthly_charges", "total_charges",
    "contract", "payment_method", "region", "support_calls", "data_usage_gb",
]


def build_preprocessor(scale: bool = True, random_state: int = 0) -> Pipeline:
    """Return Pipeline(domain features -> ColumnTransformer).

    scale=False for tree models (scaling does not change split order).
    """
    numeric = Pipeline([
        ("winsor", Winsorizer(0.01, 0.99)),  # clip before imputing: indicator columns stay 0/1
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler() if scale else "passthrough"),
    ])
    low_card = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                              sparse_output=False)),
    ])
    high_card = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        # cross-fitting inside fit_transform prevents target leakage; since scikit-learn 1.9 the
        # shuffling is controlled by passing a CV splitter (random_state/shuffle are deprecated)
        ("te", TargetEncoder(target_type="binary",
                             cv=StratifiedKFold(5, shuffle=True, random_state=random_state))),
    ])
    columns = ColumnTransformer(
        [
            ("num", numeric, NUM_COLS),
            ("low", low_card, LOW_CARD_COLS),
            ("high", high_card, HIGH_CARD_COLS),
            ("pass", "passthrough", PASSTHROUGH),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("domain", ChurnFeatures()), ("columns", columns)])
