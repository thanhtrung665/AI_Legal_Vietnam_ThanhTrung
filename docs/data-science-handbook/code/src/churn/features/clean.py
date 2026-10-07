"""Stateless cleaning: safe to run before the train/test split (it learns nothing from data)."""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

CATEGORICAL = ["contract", "payment_method", "region"]
CONTRACT_ALIASES = {"monthly": "month-to-month", "1-year": "one-year", "2-year": "two-year"}


def normalize_text(s: pd.Series) -> pd.Series:
    """Unicode NFC, trim, collapse whitespace, lower-case. Missing values become np.nan."""
    out = (
        s.astype("string")
        .map(lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )
    return out.astype(object).where(out.notna(), np.nan)


def clean_churn(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 1. Exact duplicates, then business-key duplicates (keep the latest record).
    out = out.drop_duplicates()
    out["signup_date"] = pd.to_datetime(out["signup_date"], errors="coerce")
    out = out.sort_values("signup_date").drop_duplicates("customer_id", keep="last")

    # 2. Canonical categories.
    for col in CATEGORICAL:
        if col in out:
            out[col] = normalize_text(out[col])
    if "contract" in out:
        out["contract"] = out["contract"].replace(CONTRACT_ALIASES)

    # 3. Impossible values -> NaN (imputed later inside the Pipeline). Never drop silently.
    if "age" in out:
        out["age"] = out["age"].astype(float)
        out.loc[~out["age"].between(18, 100), "age"] = np.nan
    if "monthly_charges" in out:
        out.loc[out["monthly_charges"] <= 0, "monthly_charges"] = np.nan
    return out.reset_index(drop=True)
