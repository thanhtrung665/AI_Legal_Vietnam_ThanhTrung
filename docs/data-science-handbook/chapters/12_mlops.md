# CHƯƠNG 12. MLOps — ĐƯA MÔ HÌNH VÀO PRODUCTION VÀ GIỮ NÓ ĐÚNG

> *"Chỉ một phần nhỏ của hệ thống ML thực tế là code ML."* — Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 2015).

## 12.1. Mức độ trưởng thành MLOps (theo Google Cloud)

| Mức | Đặc điểm | Dấu hiệu |
|---|---|---|
| **Level 0 — Thủ công** | Notebook → file model → gửi cho kỹ sư deploy | Không tái lập được; retrain vài tháng/lần; không giám sát |
| **Level 1 — Tự động hóa pipeline ML** | Pipeline huấn luyện tự động, có feature store, model registry, giám sát, retrain theo trigger | Continuous Training (CT) |
| **Level 2 — CI/CD cho pipeline** | Thay đổi code pipeline được test, build, deploy tự động | CI/CD/CT đầy đủ |

Mục tiêu thực tế cho phần lớn đội: **Level 1 vững chắc**.

## 12.2. Kiến trúc tham chiếu

```text
                ┌──────────── Data Sources (DB, API, Events) ────────────┐
                ▼                                                        ▼
        ┌──────────────┐  validate   ┌──────────────┐           ┌──────────────┐
        │  Ingestion   │────────────▶│ Feature      │──────────▶│ Feature Store│
        │  (Airflow/   │  (pandera/  │ Pipeline     │  offline  │ (Feast)      │
        │   Prefect)   │   GE)       └──────────────┘  + online └──────┬───────┘
        └──────────────┘                                               │
                                                                       ▼
 ┌──────────┐  push   ┌──────────────┐ train/eval ┌──────────────┐  ┌────────────────┐
 │  Git     │────────▶│  CI/CD       │───────────▶│ Training     │─▶│ Model Registry │
 │ (code,   │         │ (GitHub      │            │ Pipeline     │  │ (MLflow)       │
 │  config) │         │  Actions)    │            │ + MLflow     │  │ champion /     │
 └──────────┘         └──────────────┘            └──────────────┘  │ challenger     │
                                                                    └───────┬────────┘
                                         ┌──────────────────────────────────┼───────────┐
                                         ▼                                  ▼           │
                                ┌─────────────────┐               ┌────────────────┐    │
                                │ Batch scoring   │               │ Online serving │    │
                                │ (daily job)     │               │ (FastAPI + K8s)│    │
                                └────────┬────────┘               └───────┬────────┘    │
                                         └──────────────┬─────────────────┘             │
                                                        ▼                               │
                                          ┌───────────────────────────┐   trigger       │
                                          │ Monitoring: data drift,   │────retrain──────┘
                                          │ prediction drift, perf,   │
                                          │ latency (Evidently,       │
                                          │ Prometheus, Grafana)      │
                                          └───────────────────────────┘
```

## 12.3. Experiment Tracking & Model Registry với MLflow

```bash
# Server tracking (production: backend Postgres + artifact S3)
mlflow server --backend-store-uri postgresql://mlflow:***@db/mlflow \
              --artifacts-destination s3://ml-artifacts/mlflow --host 0.0.0.0 --port 5000
```

```python
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("churn-prediction")

# Autolog: tự động log params, metrics, model cho sklearn/lightgbm/xgboost
mlflow.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="lgbm-optuna-best", tags={"team": "ds", "data_version": "2026-10"}):
    pipe.fit(X_train, y_train)
    mlflow.log_metric("valid_pr_auc", average_precision_score(y_valid, pipe.predict_proba(X_valid)[:, 1]))
    mlflow.log_artifact("reports/figures/model_dashboard.png")
    mlflow.log_dict(study.best_params, "best_params.json")
```

### Quy trình promote mô hình: Champion / Challenger (`src/churn/models/registry.py`)

```python
client = MlflowClient()
MODEL = "churn-classifier"


def promote_if_better(candidate_version: str, metric: str = "test_pr_auc",
                      min_gain: float = 0.005) -> bool:
    """So sánh challenger với champion hiện tại; chỉ promote nếu tốt hơn rõ rệt."""
    cand = client.get_model_version(MODEL, candidate_version)
    cand_score = client.get_run(cand.run_id).data.metrics[metric]
    try:
        champ = client.get_model_version_by_alias(MODEL, "champion")
        champ_score = client.get_run(champ.run_id).data.metrics[metric]
    except Exception:                                         # chưa có champion
        champ_score = float("-inf")

    if cand_score >= champ_score + min_gain:
        client.set_registered_model_alias(MODEL, "champion", candidate_version)
        client.set_model_version_tag(MODEL, candidate_version, "validated_by", "ci-gate")
        return True
    client.set_registered_model_alias(MODEL, "challenger", candidate_version)
    return False


# Nạp mô hình theo alias — code serving không cần biết số version
model = mlflow.sklearn.load_model(f"models:/{MODEL}@champion")
```

## 12.4. Testing hệ thống ML

| Tầng | Kiểm tra gì | Công cụ |
|---|---|---|
| Unit test | Hàm làm sạch, transformer, feature | `pytest` |
| Data test | Schema, miền giá trị, null, volume, freshness | `pandera`, Great Expectations |
| Model test — hành vi | Invariance, directional, minimum functionality (CheckList, Ribeiro 2020) | `pytest` |
| Model test — chất lượng | Metric ≥ ngưỡng; không tệ hơn champion; fairness | `pytest` + MLflow |
| Integration test | Pipeline end-to-end trên dữ liệu mẫu | `pytest` |
| API test | Schema request/response, lỗi đầu vào, độ trễ | `httpx`, `locust` |

```python
# tests/unit/test_features.py
import numpy as np
import pandas as pd
import pytest

from churn.features.clean import clean_churn
from churn.features.build import Winsorizer


def test_winsorizer_clips_to_train_quantiles():
    X_train = np.arange(100, dtype=float).reshape(-1, 1)
    w = Winsorizer(0.05, 0.95).fit(X_train)
    out = w.transform(np.array([[-1000.0], [50.0], [1000.0]]))
    assert out[0, 0] == pytest.approx(np.quantile(X_train, 0.05))
    assert out[1, 0] == 50.0
    assert out[2, 0] == pytest.approx(np.quantile(X_train, 0.95))


def test_clean_normalizes_contract_and_dedups():
    raw = pd.DataFrame({
        "customer_id": ["C0000001", "C0000001"], "signup_date": ["2024-01-01", "2024-01-01"],
        "contract": [" Month-to-Month ", " Month-to-Month "], "payment_method": ["cash", "cash"],
        "region": ["R01", "R01"], "age": [30, 30], "monthly_charges": [50.0, 50.0],
    })
    out = clean_churn(raw)
    assert len(out) == 1
    assert out.loc[0, "contract"] == "month-to-month"
```

```python
# tests/model/test_model_behavior.py
import joblib
import pandas as pd
import pytest

MODEL = joblib.load("models/model.joblib")
BASE = pd.DataFrame([{
    "customer_id": "C0000001", "signup_date": pd.Timestamp("2024-01-01"), "age": 35.0,
    "tenure_months": 24, "monthly_charges": 70.0, "total_charges": 1680.0,
    "contract": "one-year", "payment_method": "credit-card", "region": "r01",
    "support_calls": 1, "data_usage_gb": 12.0,
}])


def score(**overrides) -> float:
    return float(MODEL.predict_proba(BASE.assign(**overrides))[:, 1][0])


def test_invariance_to_customer_id():
    """Đổi ID không được làm thay đổi dự đoán."""
    assert score(customer_id="C9999999") == pytest.approx(score())


def test_directional_support_calls_increase_risk():
    """Nhiều cuộc gọi khiếu nại hơn -> rủi ro không được giảm."""
    assert score(support_calls=8) >= score(support_calls=0)


def test_directional_contract():
    assert score(contract="month-to-month") > score(contract="two-year")


def test_handles_missing_and_unseen_values():
    p = score(age=None, payment_method="crypto", region="r99")
    assert 0.0 <= p <= 1.0
```

```python
# tests/model/test_quality_gate.py — chạy trong CI sau khi train
import json

THRESHOLDS = {"valid_pr_auc": 0.45, "valid_roc_auc": 0.75}


def test_quality_gate():
    metrics = json.load(open("reports/metrics.json"))
    for name, minimum in THRESHOLDS.items():
        assert metrics[name] >= minimum, f"{name}={metrics[name]:.4f} < {minimum}"
```

## 12.5. Serving

### Lựa chọn kiểu triển khai

| Kiểu | Độ trễ | Khi nào dùng | Công nghệ |
|---|---|---|---|
| **Batch** | Giờ/ngày | Danh sách gọi điện hằng ngày, báo cáo | Airflow/Prefect + Spark/pandas, ghi ra DB |
| **Online (request/response)** | ms | Chấm điểm khi khách thao tác, chống gian lận | FastAPI, BentoML, KServe, Triton |
| **Streaming** | giây | Sự kiện liên tục (clickstream) | Kafka + Flink/Spark Streaming |
| **Edge / on-device** | ms, offline | Mobile, IoT | ONNX Runtime, TFLite, Core ML |

### Batch scoring

```python
# src/churn/models/predict.py
import argparse
from datetime import date

import mlflow
import pandas as pd

from churn.data.validate import validate
from churn.features.clean import clean_churn


def batch_score(input_path: str, output_path: str, threshold: float,
                model_uri: str = "models:/churn-classifier@champion") -> None:
    model = mlflow.sklearn.load_model(model_uri)
    df = clean_churn(pd.read_parquet(input_path))
    validate(df.assign(churn=0))                       # validate input trước khi chấm điểm
    out = df[["customer_id"]].copy()
    out["score"] = model.predict_proba(df)[:, 1]
    out["is_high_risk"] = out["score"] >= threshold
    out["score_date"] = date.today().isoformat()
    out["model_uri"] = model_uri
    out.to_parquet(output_path, index=False)           # lưu để giám sát & audit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input"); p.add_argument("--output"); p.add_argument("--threshold", type=float)
    a = p.parse_args()
    batch_score(a.input, a.output, a.threshold)
```

### Online serving với FastAPI

```python
# src/churn/serving/api.py
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

log = logging.getLogger("churn-api")
MODEL_PATH = "models/model.joblib"
MODEL_VERSION = "v3"
THRESHOLD = 0.35

PREDICTIONS = Counter("churn_predictions_total", "Số dự đoán", ["model_version", "label"])
LATENCY = Histogram("churn_predict_latency_seconds", "Độ trễ dự đoán",
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)          # nạp 1 lần khi khởi động
    log.info("Model loaded: %s", MODEL_VERSION)
    yield
    state.clear()


app = FastAPI(title="Churn Prediction API", version=MODEL_VERSION, lifespan=lifespan)
app.mount("/metrics", make_asgi_app())                # Prometheus scrape endpoint


class Customer(BaseModel):
    customer_id: str = Field(pattern=r"^C\d{7}$")
    signup_date: datetime
    age: float | None = Field(None, ge=18, le=100)
    tenure_months: int = Field(ge=0, le=600)
    monthly_charges: float = Field(gt=0, le=2000)
    total_charges: float = Field(ge=0)
    contract: Literal["month-to-month", "one-year", "two-year"]
    payment_method: str | None = None
    region: str
    support_calls: int = Field(ge=0)
    data_usage_gb: float | None = Field(None, ge=0)


class Prediction(BaseModel):
    customer_id: str
    churn_probability: float
    is_high_risk: bool
    model_version: str
    request_id: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    if "model" not in state:
        raise HTTPException(503, "Model not loaded")
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=list[Prediction])
def predict(customers: list[Customer]) -> list[Prediction]:
    if not customers or len(customers) > 1000:
        raise HTTPException(422, "Batch size phải trong [1, 1000]")
    start = time.perf_counter()
    df = pd.DataFrame([c.model_dump() for c in customers])
    df["signup_date"] = pd.to_datetime(df["signup_date"]).dt.tz_localize(None)
    try:
        proba = state["model"].predict_proba(df)[:, 1]
    except Exception as exc:                          # không lộ stacktrace cho client
        log.exception("Prediction failed")
        raise HTTPException(500, "Prediction error") from exc
    LATENCY.observe(time.perf_counter() - start)

    request_id = str(uuid.uuid4())
    results = []
    for cid, p in zip(df["customer_id"], proba):
        high = bool(p >= THRESHOLD)
        PREDICTIONS.labels(MODEL_VERSION, str(high)).inc()
        results.append(Prediction(customer_id=cid, churn_probability=round(float(p), 6),
                                  is_high_risk=high, model_version=MODEL_VERSION,
                                  request_id=request_id))
    # Log input + output (không log PII) để giám sát drift & làm dữ liệu huấn luyện sau này
    log.info("request_id=%s n=%d mean_p=%.4f", request_id, len(results), float(proba.mean()))
    return results
```

```python
# tests/api/test_api.py
from fastapi.testclient import TestClient

from churn.serving.api import app

PAYLOAD = [{"customer_id": "C0000001", "signup_date": "2024-01-01T00:00:00", "age": 30,
            "tenure_months": 3, "monthly_charges": 90.0, "total_charges": 270.0,
            "contract": "month-to-month", "payment_method": "cash", "region": "r01",
            "support_calls": 4, "data_usage_gb": 5.0}]


def test_predict_ok():
    with TestClient(app) as client:                    # chạy lifespan -> nạp model
        r = client.post("/predict", json=PAYLOAD)
        assert r.status_code == 200
        assert 0 <= r.json()[0]["churn_probability"] <= 1


def test_predict_rejects_invalid_contract():
    with TestClient(app) as client:
        bad = [{**PAYLOAD[0], "contract": "lifetime"}]
        assert client.post("/predict", json=bad).status_code == 422
```

## 12.6. Đóng gói với Docker

```dockerfile
# Dockerfile — multi-stage, non-root, image nhỏ
FROM python:3.11-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir uv
COPY pyproject.toml requirements.lock ./
RUN uv pip install --system --no-cache -r requirements.lock
COPY src/ src/
RUN uv pip install --system --no-cache --no-deps .

FROM python:3.11-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
WORKDIR /app
COPY models/model.joblib models/model.joblib
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "churn.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

> `libgomp1` cần cho LightGBM. Trong hệ thống lớn, không "nướng" model vào image mà tải từ registry khi khởi động (tách vòng đời model và code).

```yaml
# docker-compose.yml — môi trường local đầy đủ
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [mlflow]
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.14.1
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlruns
    ports: ["5000:5000"]
    volumes: ["./mlruns:/mlruns"]
  prometheus:
    image: prom/prometheus
    volumes: ["./ops/prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]
  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
```

## 12.7. CI/CD với GitHub Actions

```yaml
# .github/workflows/ci.yml
name: ml-ci
on:
  pull_request:
  push:
    branches: [main]

jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: pip }
      - run: pip install -e ".[dev]"
      - run: ruff check src tests && ruff format --check src tests
      - run: mypy src
      - run: pytest tests/unit tests/data -q

  train-and-gate:
    needs: lint-test
    runs-on: ubuntu-latest
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: pip }
      - run: pip install -e ".[dev]" "dvc[s3]"
      - run: dvc pull data/raw/churn.parquet
      - run: python -m churn.models.train --config configs/train.yaml
      - run: pytest tests/model -q            # behavior tests + quality gate
      - uses: actions/upload-artifact@v4
        with: { name: model, path: models/ }

  build-push:
    if: github.ref == 'refs/heads/main'
    needs: train-and-gate
    runs-on: ubuntu-latest
    permissions: { contents: read, packages: write }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with: { name: model, path: models/ }
      - uses: docker/login-action@v3
        with: { registry: ghcr.io, username: "${{ github.actor }}", password: "${{ secrets.GITHUB_TOKEN }}" }
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}/churn-api:${{ github.sha }}
```

## 12.8. Chiến lược triển khai an toàn

| Chiến lược | Cách làm | Ưu | Nhược |
|---|---|---|---|
| **Shadow** | Mô hình mới nhận traffic thật nhưng kết quả không được dùng; so sánh với mô hình cũ | Không rủi ro cho người dùng | Tốn tài nguyên; không đo được hiệu quả kinh doanh |
| **Canary** | 5% → 25% → 100% traffic, theo dõi metric từng bước | Giới hạn thiệt hại | Cần hạ tầng định tuyến |
| **Blue/Green** | Hai môi trường song song, chuyển traffic tức thời | Rollback nhanh | Gấp đôi tài nguyên |
| **A/B test** | Chia ngẫu nhiên người dùng, đo KPI kinh doanh | Đo được tác động nhân quả | Cần thời gian đủ cỡ mẫu (chương 4.5) |
| **Multi-armed bandit** | Tự động dồn traffic cho mô hình tốt hơn | Ít "chi phí cơ hội" | Phức tạp hơn để phân tích |

**Luôn có kế hoạch rollback và fallback** (ví dụ: rule-based baseline khi mô hình lỗi hoặc timeout).

## 12.9. Giám sát (Monitoring)

### Các loại suy giảm mô hình

| Loại | Định nghĩa | Ví dụ | Phát hiện |
|---|---|---|---|
| **Data drift (covariate shift)** | $P(X)$ thay đổi | Khách hàng trẻ hơn sau chiến dịch marketing | PSI, KS, χ², Jensen-Shannon |
| **Prior/label shift** | $P(Y)$ thay đổi | Tỷ lệ churn tăng do đối thủ giảm giá | Theo dõi tỷ lệ dự đoán dương / tỷ lệ nhãn |
| **Concept drift** | $P(Y \mid X)$ thay đổi | Quan hệ giữa giá và churn đổi sau quy định mới | Metric hiệu năng khi có nhãn |
| **Data quality issue** | Pipeline lỗi | Cột bị null toàn bộ, đổi đơn vị | Data tests, tỷ lệ null, volume |
| **Training-serving skew** | Feature tính khác nhau giữa train và serve | Code feature viết 2 lần bằng 2 ngôn ngữ | So sánh phân phối feature log tại serving vs train |

### Population Stability Index (PSI)

$$PSI = \sum_{i=1}^{B}(A_i - E_i)\ln\frac{A_i}{E_i}$$

với $E_i$, $A_i$ là tỷ lệ mẫu trong bin *i* của tập tham chiếu (train) và hiện tại.

| PSI | Diễn giải |
|---|---|
| < 0.1 | Ổn định |
| 0.1 – 0.25 | Thay đổi vừa — theo dõi |
| > 0.25 | Thay đổi lớn — điều tra, cân nhắc retrain |

```python
# src/churn/monitoring/drift.py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + eps
    a = np.histogram(actual, edges)[0] / len(actual) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def categorical_psi(expected: pd.Series, actual: pd.Series, eps: float = 1e-6) -> float:
    cats = set(expected.dropna()) | set(actual.dropna())
    e = expected.value_counts(normalize=True).reindex(cats, fill_value=0) + eps
    a = actual.value_counts(normalize=True).reindex(cats, fill_value=0) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in num_cols:
        r, k = reference[c].dropna(), current[c].dropna()
        rows.append({"feature": c, "type": "num", "psi": psi(r, k),
                     "ks_pvalue": stats.ks_2samp(r, k).pvalue,
                     "null_rate_ref": reference[c].isna().mean(),
                     "null_rate_cur": current[c].isna().mean()})
    for c in cat_cols:
        cats = sorted(set(reference[c].dropna()) | set(current[c].dropna()))
        p = reference[c].value_counts(normalize=True).reindex(cats, fill_value=0)
        q = current[c].value_counts(normalize=True).reindex(cats, fill_value=0)
        rows.append({"feature": c, "type": "cat", "psi": categorical_psi(reference[c], current[c]),
                     "js_distance": float(jensenshannon(p, q, base=2)),
                     "null_rate_ref": reference[c].isna().mean(),
                     "null_rate_cur": current[c].isna().mean(),
                     "new_categories": sorted(set(current[c].dropna()) - set(reference[c].dropna()))})
    out = pd.DataFrame(rows)
    out["status"] = pd.cut(out["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["ok", "warn", "alert"])
    return out.sort_values("psi", ascending=False)
```

> Với dữ liệu rất lớn, KS test gần như luôn cho p-value ≈ 0 dù khác biệt không đáng kể → dựa vào **độ lớn khác biệt** (PSI, khoảng cách Wasserstein, JS) hơn là p-value.

### Evidently — báo cáo drift tự động

```python
# evidently >= 0.7
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

report = Report([DataDriftPreset(), DataSummaryPreset()])
snapshot = report.run(current_data=current_df, reference_data=reference_df)
snapshot.save_html("reports/drift_report.html")

# evidently < 0.7 (API cũ):
# from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset
# r = Report(metrics=[DataDriftPreset()]); r.run(reference_data=ref, current_data=cur)
# r.save_html("drift.html")
```

### Bảng giám sát production đề xuất

| Nhóm | Metric | Ngưỡng cảnh báo ví dụ | Tần suất |
|---|---|---|---|
| Hệ thống | p95 latency, error rate, throughput, CPU/RAM | p95 > 200ms; lỗi > 1% | Real-time |
| Dữ liệu | Null rate, schema, volume, category mới | Null tăng > 5 điểm %; volume ±30% | Mỗi batch |
| Drift input | PSI từng feature (top features theo importance) | PSI > 0.25 | Hằng ngày/tuần |
| Drift output | Phân phối điểm, tỷ lệ dự đoán dương | PSI điểm > 0.1 | Hằng ngày |
| Hiệu năng | PR-AUC, recall, calibration khi có nhãn (trễ 30 ngày) | Giảm > 10% so với baseline | Khi nhãn về |
| Kinh doanh | Tỷ lệ giữ chân nhóm được liên hệ vs control | — | Hằng tháng |

## 12.10. Retraining & Orchestration

| Chiến lược retrain | Mô tả | Phù hợp |
|---|---|---|
| Định kỳ (scheduled) | Hằng tuần/tháng | Đơn giản, dễ vận hành — **mặc định tốt** |
| Theo trigger | Khi drift/hiệu năng vượt ngưỡng | Môi trường thay đổi bất thường |
| Online / incremental | Cập nhật liên tục | Gợi ý, quảng cáo, dữ liệu streaming |

Quy trình retrain **không bao giờ tự động deploy mù quáng**: train → validate dữ liệu → evaluate trên out-of-time test → so sánh champion (gate) → shadow/canary → promote.

```python
# flows/retrain.py — Prefect 2/3
from prefect import flow, task, get_run_logger


@task(retries=3, retry_delay_seconds=60)
def ingest(snapshot: str) -> str:
    ...                                   # trả về đường dẫn dữ liệu
    return f"data/raw/churn_{snapshot}.parquet"


@task
def validate_data(path: str) -> str:
    import pandas as pd
    from churn.data.validate import validate
    validate(pd.read_parquet(path))
    return path


@task
def train(path: str) -> str:
    ...                                   # gọi churn.models.train, trả về model version
    return "12"


@task
def gate_and_promote(version: str) -> bool:
    from churn.models.registry import promote_if_better
    return promote_if_better(version)


@flow(name="churn-retrain")
def retrain_flow(snapshot: str):
    log = get_run_logger()
    version = train(validate_data(ingest(snapshot)))
    promoted = gate_and_promote(version)
    log.info("Model v%s promoted=%s", version, promoted)


if __name__ == "__main__":
    retrain_flow.serve(name="monthly", cron="0 2 1 * *", parameters={"snapshot": "latest"})
```

```python
# dags/churn_retrain.py — Airflow 2.x (TaskFlow API)
import pendulum
from airflow.decorators import dag, task


@dag(schedule="0 2 1 * *", start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
     catchup=False, tags=["ml", "churn"], default_args={"retries": 2})
def churn_retrain():
    @task
    def ingest(ds=None) -> str: ...

    @task
    def validate(path: str) -> str: ...

    @task
    def train(path: str) -> str: ...

    @task
    def promote(version: str) -> None: ...

    promote(train(validate(ingest())))


churn_retrain()
```

## 12.11. Feature Store — chống training-serving skew

Feature store (Feast, Tecton, Databricks/Vertex/SageMaker Feature Store) đảm bảo **cùng một định nghĩa feature** cho cả huấn luyện (offline, point-in-time join) và serving (online, độ trễ thấp).

```python
# feature_repo/features.py — Feast
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

customer = Entity(name="customer", join_keys=["customer_id"])

usage_source = FileSource(path="data/processed/usage_features.parquet",
                          timestamp_field="event_timestamp")

usage_fv = FeatureView(
    name="customer_usage",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[Field(name="support_calls_90d", dtype=Int64),
            Field(name="avg_bill_6m", dtype=Float32)],
    source=usage_source,
)

# Training: store.get_historical_features(entity_df=labels_with_timestamps, features=[...])
# Serving : store.get_online_features(features=[...], entity_rows=[{"customer_id": "C0000001"}])
```

## 12.12. Tối ưu hiệu năng serving

- **ONNX / Treelite:** chuyển mô hình sang định dạng tối ưu (`skl2onnx`, `onnxmltools`), nhanh hơn 2–10 lần.
- **Batch inference** thay vì từng dòng; vectorize tiền xử lý.
- **Cache** kết quả cho input lặp lại; **pre-compute** điểm offline khi có thể.
- **Scale ngang** bằng Kubernetes HPA theo CPU/RPS; GPU chỉ khi cần (deep learning).

## 12.13. Bảo mật, quyền riêng tư & quản trị

- Secret qua biến môi trường / secret manager (Vault, AWS Secrets Manager) — **không bao giờ** commit vào git.
- Ẩn danh hóa/hash PII trước khi đưa vào pipeline huấn luyện; giới hạn quyền truy cập theo vai trò.
- Ghi lại **lineage**: mô hình version X được train từ dữ liệu version Y, code commit Z, config W.
- Audit log mọi dự đoán quan trọng (input hash, output, model version, thời gian).
- Đánh giá rủi ro AI theo khung pháp lý áp dụng (Nghị định 13/2023/NĐ-CP về dữ liệu cá nhân; EU AI Act nếu liên quan).
- Kiểm tra lỗ hổng phụ thuộc (`pip-audit`), quét image (`trivy`).

> **Checklist Chương 12**
> - [ ] Mọi thí nghiệm được log (params, metrics, artifact, data version, git commit).
> - [ ] Model registry với alias champion/challenger; promote qua quality gate tự động.
> - [ ] Có unit test, data test, behavior test, quality gate trong CI.
> - [ ] API có validate input (pydantic), health/ready, metrics, logging, xử lý lỗi.
> - [ ] Docker image non-root, phiên bản cố định; CI/CD build & deploy tự động.
> - [ ] Triển khai qua shadow/canary; có rollback & fallback.
> - [ ] Giám sát hệ thống, dữ liệu, drift, hiệu năng, KPI kinh doanh với ngưỡng cảnh báo.
> - [ ] Có chiến lược retrain và runbook xử lý sự cố.
