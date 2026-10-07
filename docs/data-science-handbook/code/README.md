# `churn` — reference implementation for the handbook

Runnable, tested code behind the handbook chapters (Python ≥ 3.11).

```bash
cd docs/data-science-handbook/code
pip install -e ".[dev,extra]"
pytest -q                                            # unit, data-contract, split, behaviour, API tests
python -m churn.models.train --config configs/train.yaml   # trains, logs to MLflow (sqlite:///mlflow.db)
make serve                                           # FastAPI on :8000
```

| Module | Chapter |
|---|---|
| `churn.data.synthetic` | 0 — dataset used throughout |
| `churn.data.validate` | 2 — data contract (pandera) |
| `churn.features.clean`, `churn.features.transformers`, `churn.features.build` | 5–6 |
| `churn.data.split` | 7 |
| `churn.models.train`, `churn.models.registry` | 8, 12 |
| `churn.models.evaluate` | 9 |
| `churn.monitoring.drift` | 12 |
| `churn.serving.api` | 12 |
| `churn.viz.style` | 11 |
