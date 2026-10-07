"""Drift statistics computed against a frozen reference (training) sample."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon


def psi(expected, actual, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index with quantile bins learned on the reference sample."""
    e_arr, a_arr = np.asarray(expected, float), np.asarray(actual, float)
    e_arr, a_arr = e_arr[~np.isnan(e_arr)], a_arr[~np.isnan(a_arr)]
    edges = np.unique(np.quantile(e_arr, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(e_arr, edges)[0] / len(e_arr) + eps
    a = np.histogram(a_arr, edges)[0] / len(a_arr) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def categorical_psi(expected: pd.Series, actual: pd.Series, eps: float = 1e-6) -> float:
    cats = sorted(set(expected.dropna()) | set(actual.dropna()))
    e = expected.value_counts(normalize=True).reindex(cats, fill_value=0) + eps
    a = actual.value_counts(normalize=True).reindex(cats, fill_value=0) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, num_cols: list[str],
                 cat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in num_cols:
        r, k = reference[c].dropna(), current[c].dropna()
        rows.append({
            "feature": c, "type": "num", "psi": psi(r, k),
            "ks_stat": stats.ks_2samp(r, k).statistic,
            "wasserstein_std": stats.wasserstein_distance(r, k) / (r.std() or 1.0),
            "null_ref": reference[c].isna().mean(), "null_cur": current[c].isna().mean(),
        })
    for c in cat_cols:
        cats = sorted(set(reference[c].dropna()) | set(current[c].dropna()))
        p = reference[c].value_counts(normalize=True).reindex(cats, fill_value=0)
        q = current[c].value_counts(normalize=True).reindex(cats, fill_value=0)
        rows.append({
            "feature": c, "type": "cat", "psi": categorical_psi(reference[c], current[c]),
            "js_distance": float(jensenshannon(p, q, base=2)),
            "null_ref": reference[c].isna().mean(), "null_cur": current[c].isna().mean(),
            "new_categories": sorted(set(current[c].dropna()) - set(reference[c].dropna())),
        })
    out = pd.DataFrame(rows)
    out["status"] = pd.cut(out["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["ok", "warn", "alert"])
    return out.sort_values("psi", ascending=False, ignore_index=True)
