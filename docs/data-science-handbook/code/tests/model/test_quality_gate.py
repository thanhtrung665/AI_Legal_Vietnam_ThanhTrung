from sklearn.metrics import average_precision_score, roc_auc_score

from churn.features.build import RAW_FEATURES


def test_model_beats_minimum_quality(trained_model):
    from churn.data.synthetic import make_churn_data
    from churn.features.clean import clean_churn

    holdout = clean_churn(make_churn_data(n=4000, seed=99))
    p = trained_model.predict_proba(holdout[RAW_FEATURES])[:, 1]
    assert roc_auc_score(holdout["churn"], p) > 0.70
    # PR-AUC baseline = prevalence; require a clear lift over a random ranking
    assert average_precision_score(holdout["churn"], p) > 1.6 * holdout["churn"].mean()
