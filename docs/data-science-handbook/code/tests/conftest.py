import pytest

from churn.data.synthetic import make_churn_data
from churn.features.clean import clean_churn


@pytest.fixture(scope="session")
def clean_df():
    return clean_churn(make_churn_data(n=4000, seed=7))


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory, clean_df):
    """Train once per test session on a small sample (fast, deterministic)."""
    import lightgbm as lgb
    from sklearn.pipeline import Pipeline

    from churn.features.build import RAW_FEATURES, build_preprocessor

    clf = lgb.LGBMClassifier(n_estimators=150, num_leaves=15, random_state=0, verbose=-1)
    model = Pipeline([("prep", build_preprocessor(scale=False)), ("clf", clf)])
    return model.fit(clean_df[RAW_FEATURES], clean_df["churn"])
