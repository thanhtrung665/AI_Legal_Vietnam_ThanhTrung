# CHƯƠNG 12. MLOps — ĐƯA MÔ HÌNH VÀO PRODUCTION VÀ GIỮ NÓ ĐÚNG

> *"Only a small fraction of real-world ML systems is composed of the ML code."* — Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 2015).

**Mục tiêu chương:** biến mô hình trong notebook thành **hệ thống** tái lập được, kiểm thử được, triển khai an toàn, được giám sát và tự cải thiện. Toàn bộ ví dụ dùng package tham chiếu `code/src/churn`, đã được chạy kiểm chứng.

## 12.1. Vì sao MLOps: nợ kỹ thuật đặc thù của ML

Sculley et al. (2015) chỉ ra các dạng nợ kỹ thuật **chỉ có ở hệ thống ML**:

| Dạng nợ | Mô tả | Biện pháp |
|---|---|---|
| **CACE** (*Changing Anything Changes Everything*) | Feature/tham số đan xen, sửa một chỗ ảnh hưởng mọi chỗ | Kiểm thử hành vi, so sánh champion/challenger |
| **Hidden feedback loops** | Dự đoán ảnh hưởng dữ liệu tương lai: khách được gọi thì không churn, nên nhãn bị lệch | Nhóm đối chứng (holdout) cố định |
| **Undeclared consumers** | Hệ thống khác âm thầm dùng output mô hình | Access control, hợp đồng API, versioning |
| **Data dependency debt** | Phụ thuộc dữ liệu không ổn định hoặc không cần thiết | Data contract, loại feature ít giá trị |
| **Glue code / pipeline jungles** | Code dán nối, pipeline chắp vá | Package hóa, pipeline khai báo (DAG) |
| **Configuration debt** | Cấu hình rải rác, không review | Config có kiểu, trong git, có test |
| **Training–serving skew** | Feature tính khác nhau giữa huấn luyện và phục vụ | Dùng chung code/pipeline, feature store |

### Mức độ trưởng thành MLOps (Google Cloud Architecture Center)

| Mức | Đặc điểm | Dấu hiệu |
|---|---|---|
| **Level 0: Thủ công** | Notebook → file model → bàn giao cho kỹ sư; retrain hiếm | Không tái lập; không giám sát; "mô hình chạy được trên máy tôi" |
| **Level 1: Tự động hóa pipeline ML** | Pipeline huấn luyện tự động, **continuous training** (CT) theo lịch/trigger; data & model validation; feature store; metadata | Mô hình mới tự được huấn luyện và kiểm định |
| **Level 2: CI/CD cho pipeline** | Thay đổi code pipeline được test, build, deploy tự động | Thử nghiệm nhanh ý tưởng mới một cách an toàn |

Mục tiêu thực tế cho phần lớn đội: **Level 1 vững chắc**.

## 12.2. Kiến trúc tham chiếu

```text
  Nguồn dữ liệu ──▶ Ingestion + Data contract (pandera) ──▶ Feature pipeline ──▶ Offline store / Feature store
                                                                                       │
  Git (code, config) ──▶ CI: lint, unit/data/model tests ──▶ Training pipeline ◀──────┘
                                                              │  (MLflow: params, metrics, model, lineage)
                                                              ▼
                                                    Model Registry (alias champion / challenger)
                                                              │  quality gate + approval
                                     ┌────────────────────────┴───────────────────────┐
                                     ▼                                                ▼
                         Batch scoring (lịch hằng ngày)                 Online API (FastAPI, K8s, autoscale)
                                     └──────────────┬─────────────────────────────────┘
                                                    ▼
                     Monitoring: hệ thống · dữ liệu · drift · hiệu năng (nhãn trễ) · KPI kinh doanh
                                                    │  vượt ngưỡng
                                                    └────────▶ Retraining trigger (CT) ──▶ Training pipeline
```

### Thiết lập

```python
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
import churn  # noqa: E402

CODE_DIR = Path(churn.__file__).resolve().parents[2]          # docs/data-science-handbook/code
print("Package churn:", churn.__version__, "| code dir:", CODE_DIR.name)
```

## 12.3. Kiểm thử hệ thống ML

### ML Test Score (Breck et al., 2017, Google)

Rubric gồm **28 kiểm thử** chia 4 nhóm. Điểm của hệ thống bằng điểm **thấp nhất** trong 4 nhóm, vì hệ thống chỉ mạnh bằng mắt xích yếu nhất. Mỗi kiểm thử được 0.5 điểm nếu làm thủ công và 1 điểm nếu tự động.

| Nhóm | Ví dụ kiểm thử (trích) |
|---|---|
| **Features & Data** | Kỳ vọng phân phối feature được ghi thành schema; mọi feature đều có ích (không thừa); chi phí của feature được đo; tuân thủ yêu cầu meta-level (PII); code tạo feature có unit test |
| **Model Development** | Mọi thay đổi mô hình được review và lưu trong VCS; metric offline tương quan với metric online; mọi siêu tham số đã được tune; độ "cũ" của mô hình được biết; mô hình được so với **baseline đơn giản**; chất lượng được kiểm tra theo **phân khúc** |
| **ML Infrastructure** | Huấn luyện tái lập được; mô hình được unit test; pipeline được integration test end-to-end; chất lượng được kiểm định **trước khi** serve; mô hình có thể debug từng ví dụ; **canary** trước khi phục vụ toàn bộ; **rollback** được |
| **Monitoring** | Phát hiện thay đổi dependency; dữ liệu đầu vào ổn định; **training–serving skew**; mô hình không quá cũ; không có NaN/Inf trong output; không suy giảm hiệu năng tính toán; không suy giảm chất lượng dự đoán |

### Các tầng kiểm thử trong package tham chiếu

| Tầng | File | Kiểm tra gì |
|---|---|---|
| Unit | `tests/unit/test_transformers.py` | Winsorizer học đúng ngưỡng, giữ NaN, qua `check_transformer_general` của sklearn; làm sạch chuỗi; dedup |
| Data | `tests/data/test_contract.py`, `test_split.py` | Contract bắt đúng lỗi; split theo thời gian không chồng lấn; phát hiện leakage |
| Model: hành vi | `tests/model/test_behavior.py` | **Invariance** (đổi ID không đổi dự đoán), **directional** (nhiều khiếu nại thì rủi ro tăng), **robustness** (giá trị lạ/thiếu) |
| Model: chất lượng | `tests/model/test_quality_gate.py` | AUC, PR-AUC vượt ngưỡng tối thiểu |
| API | `tests/api/test_api.py` | Schema, mã lỗi 422, `/ready` |

```python norun
# tests/model/test_behavior.py (trích) — kiểm thử hành vi theo CheckList (Ribeiro et al., 2020)
def test_invariance_to_customer_id(score):
    assert score(customer_id="C9999999") == pytest.approx(score())


def test_directional_support_calls(score):
    assert score(support_calls=8) > score(support_calls=0)


def test_robust_to_missing_and_unseen(score):
    assert 0.0 <= score(age=None, payment_method="crypto", region="r99") <= 1.0
```

Chạy toàn bộ bộ test của package (CI làm đúng lệnh này):

```python
result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=CODE_DIR,
                        capture_output=True, text=True, env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"})
print(result.stdout.strip().splitlines()[-1])
assert result.returncode == 0
```

## 12.4. Experiment tracking & Model Registry (MLflow 3)

**[Docs]** Các khái niệm của MLflow 3:

- **Experiment / Run:** một lần huấn luyện, gồm params, metrics, tags, artifacts.
- **LoggedModel:** mô hình là thực thể hạng nhất, có `model_id`, được liên kết với run, metrics và dataset. Nạp bằng `models:/<model_id>`.
- **Registered Model + Version:** phiên bản được quản trị trong **Model Registry**.
- **Alias** (từ MLflow 2.9, thay cho *stages* Staging/Production đã deprecate): nhãn di động như `champion`, `challenger`. Code serving nạp `models:/churn-classifier@champion` mà **không cần biết số version**.
- **Backend:** file store `./mlruns` ở chế độ bảo trì; dùng database (SQLite cho local, PostgreSQL/MySQL cho production) cùng artifact store (S3/GCS/Azure Blob).

```bash
# Tracking server production: metadata trong PostgreSQL, artifact trong S3
mlflow server --backend-store-uri postgresql://mlflow:***@db:5432/mlflow \
              --artifacts-destination s3://ml-artifacts/mlflow --host 0.0.0.0 --port 5000
```

### Quy trình champion/challenger chạy thật

```python
import mlflow
from mlflow import MlflowClient

from churn.models.registry import promote_if_better
from churn.models.train import main as train_main

Path("configs").mkdir(exist_ok=True)
base_cfg = """
target: churn
random_state: 42
split: {date_col: signup_date, train_end: "2023-10-01", valid_end: "2024-01-01", gap_days: 0}
model_params: {n_estimators: %d, learning_rate: 0.03, num_leaves: %d, min_child_samples: 50,
               subsample: 0.8, subsample_freq: 1, colsample_bytree: 0.8, reg_lambda: 1.0}
"""
client = MlflowClient()
MODEL = "churn-classifier"
history = []
for version_cfg in [(60, 3), (400, 15), (400, 63)]:                # 3 lần huấn luyện với cấu hình khác nhau
    Path("configs/train.yaml").write_text(base_cfg % version_cfg)
    metrics = train_main("configs/train.yaml", out_dir=".")
    run = mlflow.search_runs(experiment_names=["churn-prediction"], order_by=["start_time DESC"],
                             max_results=1).iloc[0]
    model_id = mlflow.search_logged_models(experiment_ids=[run["experiment_id"]], order_by=[
        {"field_name": "creation_timestamp", "ascending": False}], max_results=1, output_format="list")[0].model_id
    mv = mlflow.register_model(f"models:/{model_id}", MODEL)
    promoted = promote_if_better(MODEL, mv.version, metric="valid_pr_auc", min_gain=0.005, client=client)
    history.append({"version": mv.version, "cfg(n_estimators,num_leaves)": version_cfg,
                    "valid_pr_auc": round(metrics["valid_pr_auc"], 4), "promoted": promoted})
print(pd.DataFrame(history).to_string(index=False))

champion = client.get_model_version_by_alias(MODEL, "champion")
print("Champion hiện tại: version", champion.version)
served = mlflow.sklearn.load_model(f"models:/{MODEL}@champion")    # serving chỉ cần alias
print(type(served).__name__, "đã nạp từ registry")
```

`promote_if_better` (trong `churn/models/registry.py`) chỉ gán alias `champion` khi phiên bản mới **vượt champion hiện tại ít nhất `min_gain`**. Nếu không, nó gán `challenger` để tiếp tục chạy shadow/A/B. Trong tổ chức lớn, bước promote cần thêm **phê duyệt thủ công** và ghi lại lý do (audit trail).

## 12.5. Serving: batch, online, streaming

| Kiểu | Độ trễ | Khi nào | Công nghệ |
|---|---|---|---|
| **Batch** | Giờ/ngày | Danh sách gọi điện hằng ngày, báo cáo, BI | Airflow/Prefect + pandas/Spark, ghi ra bảng |
| **Online (request/response)** | ms | Chấm điểm khi khách thao tác, chống gian lận | FastAPI, BentoML, KServe, Triton, Seldon |
| **Streaming** | giây | Sự kiện liên tục | Kafka + Flink/Spark Structured Streaming |
| **Edge / on-device** | ms, offline | Mobile, IoT | ONNX Runtime, TFLite, Core ML |

**[Kinh nghiệm]** Nếu nghiệp vụ chấp nhận được độ trễ, **ưu tiên batch**. Batch đơn giản hơn nhiều: không cần SLA độ trễ, dễ giám sát, dễ rollback, và có thể dùng mô hình nặng.

### Batch scoring

```python
import joblib

from churn.data.synthetic import make_churn_data
from churn.data.validate import validate
from churn.features.build import RAW_FEATURES
from churn.features.clean import clean_churn


def batch_score(raw: pd.DataFrame, model, threshold: float, model_ref: str) -> pd.DataFrame:
    """Làm sạch → kiểm tra contract → chấm điểm → output có metadata để audit và giám sát."""
    clean = clean_churn(raw)
    validate(clean.assign(churn=clean.get("churn", 0)))           # contract chạy cả lúc chấm điểm
    out = clean[["customer_id"]].copy()
    out["score"] = model.predict_proba(clean[RAW_FEATURES])[:, 1]
    assert np.isfinite(out["score"]).all(), "Output có NaN/Inf"     # ML Test Score: monitoring
    out["is_high_risk"] = out["score"] >= threshold
    out["score_date"] = pd.Timestamp.today().normalize()
    out["model_ref"] = model_ref
    return out


scores = batch_score(make_churn_data(n=5_000, seed=7), served, threshold=0.33,
                     model_ref=f"{MODEL}@champion(v{champion.version})")
scores.to_parquet("scores_latest.parquet", index=False)
print(scores.head(3))
print("Tỷ lệ khách nguy cơ cao:", round(scores["is_high_risk"].mean(), 3))
```

### Online serving với FastAPI

Mã nguồn: `code/src/churn/serving/api.py`. Các điểm production:

- **Nạp mô hình một lần** trong `lifespan` (không nạp mỗi request).
- **Validate input** bằng **pydantic v2**: kiểu, miền giá trị, `Literal` cho phân loại. Input sai trả về **422** tự động.
- **`/health` (liveness)** và **`/ready` (readiness)** riêng biệt cho Kubernetes.
- Endpoint **`def`** (không phải `async def`) cho suy luận CPU-bound. **[Docs]** FastAPI chạy hàm `def` trong threadpool, nhờ đó không chặn event loop.
- Không trả stack trace cho client; log có `request_id`; không log PII.
- Cấu hình (đường dẫn mô hình, ngưỡng, version) đọc từ **biến môi trường**.

```python
import joblib
from fastapi.testclient import TestClient

joblib.dump(served, "model.joblib")
os.environ.update({"MODEL_PATH": "model.joblib", "MODEL_VERSION": f"v{champion.version}"})
import importlib

import churn.serving.api as api

importlib.reload(api)                                             # đọc lại biến môi trường

payload = [{"customer_id": "C0000001", "signup_date": "2024-03-01T00:00:00", "age": 29,
            "tenure_months": 2, "monthly_charges": 95.0, "total_charges": 190.0, "contract": "month-to-month",
            "payment_method": "cash", "region": "r03", "support_calls": 5, "data_usage_gb": 4.2}]
with TestClient(api.app) as client_api:                           # chạy lifespan: nạp mô hình
    print(client_api.get("/ready").json())
    r = client_api.post("/predict", json=payload)
    print(r.status_code, r.json()[0])
    bad = client_api.post("/predict", json=[{**payload[0], "contract": "lifetime"}])
    print("Input sai ->", bad.status_code, bad.json()["detail"][0]["msg"])
    t0 = time.perf_counter()
    for _ in range(50):
        client_api.post("/predict", json=payload * 20)
    print(f"Độ trễ trung bình (batch 20 khách, in-process): {(time.perf_counter() - t0) / 50 * 1000:.1f} ms")
```

## 12.6. Đóng gói: Docker và Kubernetes

### Dockerfile (trong package tham chiếu)

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app          # không chạy bằng root
COPY --from=builder /install /usr/local
WORKDIR /app
COPY models/model.joblib models/model.joblib
ENV MODEL_PATH=/app/models/model.joblib PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "churn.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

| Thực hành tốt | Lý do |
|---|---|
| Multi-stage build, image `slim` | Image nhỏ, ít bề mặt tấn công |
| Ghim phiên bản base image và lock dependency | Tái lập; tránh thay đổi âm thầm |
| User không phải root (`USER app`) | Bảo mật |
| `libgomp1` | OpenMP runtime cho LightGBM/XGBoost |
| Không "nướng" secret vào image | Secret qua env/secret manager lúc chạy |
| Cân nhắc **không** nướng model vào image | Tách vòng đời model và code: nạp từ registry khi khởi động |
| Quét lỗ hổng (`trivy`, `pip-audit`), SBOM | Chuỗi cung ứng phần mềm |

### Kubernetes: Deployment, probes, resources, autoscaling

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: {name: churn-api, labels: {app: churn-api}}
spec:
  replicas: 3
  selector: {matchLabels: {app: churn-api}}
  strategy: {type: RollingUpdate, rollingUpdate: {maxUnavailable: 0, maxSurge: 1}}
  template:
    metadata: {labels: {app: churn-api}}
    spec:
      containers:
        - name: api
          image: ghcr.io/org/churn-api:1f2e3d4            # tag bất biến (git SHA), không dùng :latest
          ports: [{containerPort: 8000}]
          env:
            - {name: MODEL_VERSION, value: "v12"}
            - {name: DECISION_THRESHOLD, value: "0.33"}
            - {name: OMP_NUM_THREADS, value: "1"}           # khớp limits.cpu (xem Chương 8.11)
          resources:
            requests: {cpu: "500m", memory: "512Mi"}
            limits: {cpu: "1", memory: "1Gi"}
          readinessProbe: {httpGet: {path: /ready, port: 8000}, initialDelaySeconds: 5, periodSeconds: 10}
          livenessProbe: {httpGet: {path: /health, port: 8000}, initialDelaySeconds: 15, periodSeconds: 20}
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: {name: churn-api}
spec:
  scaleTargetRef: {apiVersion: apps/v1, kind: Deployment, name: churn-api}
  minReplicas: 3
  maxReplicas: 20
  metrics: [{type: Resource, resource: {name: cpu, target: {type: Utilization, averageUtilization: 65}}}]
```

> **Bài học từ Chương 8.11:** trong container, số luồng OpenMP mặc định bằng số vCPU **của node**, không phải `limits.cpu` của pod. Không đặt `OMP_NUM_THREADS` sẽ gây oversubscription và độ trễ tăng hàng chục lần.

## 12.7. CI/CD/CT với GitHub Actions

```yaml
name: churn-ci
on:
  pull_request:
  push: {branches: [main]}
jobs:
  test:                                   # CI: mọi PR
    runs-on: ubuntu-latest
    defaults: {run: {working-directory: docs/data-science-handbook/code}}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12", cache: pip}
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: pytest -q                    # unit + data + behaviour + API
  train-and-gate:                         # CT: huấn luyện + quality gate
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12", cache: pip}
      - run: pip install -e "docs/data-science-handbook/code[dev]"
      - run: python -m churn.models.train --config docs/data-science-handbook/code/configs/train.yaml
      - run: python -m churn.models.promote --model churn-classifier --metric valid_pr_auc --min-gain 0.005
  build-and-deploy:                       # CD: chỉ khi có champion mới
    needs: train-and-gate
    runs-on: ubuntu-latest
    permissions: {contents: read, packages: write, id-token: write}
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v6
        with: {context: docs/data-science-handbook/code, push: true,
               tags: "ghcr.io/${{ github.repository }}/churn-api:${{ github.sha }}"}
      - run: kubectl set image deployment/churn-api api=ghcr.io/${{ github.repository }}/churn-api:${{ github.sha }}
```

> `churn.models.promote` là CLI bọc quanh `promote_if_better`. Package tham chiếu cung cấp hàm, còn CLI là phần bài tập mở rộng.

## 12.8. Chiến lược triển khai an toàn

| Chiến lược | Cách làm | Ưu | Nhược |
|---|---|---|---|
| **Shadow** | Mô hình mới nhận traffic thật, kết quả **không** được dùng; so với champion | Không rủi ro cho người dùng | Không đo được tác động kinh doanh; tốn tài nguyên |
| **Canary** | 1% → 5% → 25% → 100% traffic, dừng nếu metric xấu | Giới hạn thiệt hại | Cần định tuyến và giám sát tốt |
| Blue/Green | Hai môi trường song song, chuyển traffic tức thì | Rollback tức thì | Gấp đôi tài nguyên |
| **A/B test** | Ngẫu nhiên hóa người dùng, đo OEC (Chương 4.6) | Đo được **tác động nhân quả** | Cần thời gian và cỡ mẫu |
| Interleaving | Trộn kết quả hai mô hình trong cùng một danh sách | Nhạy hơn A/B cho ranking | Chỉ cho xếp hạng |
| Multi-armed bandit | Tự dồn traffic cho mô hình tốt hơn | Ít chi phí cơ hội | Khó phân tích thống kê |

```python
def shadow_compare(champion, challenger, X: pd.DataFrame, threshold: float) -> dict:
    """Shadow mode: champion phục vụ, challenger chỉ được log; so sánh mức độ bất đồng."""
    p_c, p_n = champion.predict_proba(X)[:, 1], challenger.predict_proba(X)[:, 1]
    flip = (p_c >= threshold) != (p_n >= threshold)
    return {"rank_corr": float(pd.Series(p_c).corr(pd.Series(p_n), method="spearman")),
            "decision_flip_rate": float(flip.mean()), "mean_score_shift": float(p_n.mean() - p_c.mean())}


challenger_v = client.get_model_version(MODEL, history[-1]["version"])
challenger = mlflow.sklearn.load_model(f"models:/{MODEL}/{challenger_v.version}")
print(shadow_compare(served, challenger, clean_churn(make_churn_data(n=3_000, seed=11))[RAW_FEATURES], 0.33))
```

**Luôn có kế hoạch rollback** (đổi alias `champion` về version cũ) và **fallback** (quy tắc baseline) khi mô hình lỗi hoặc timeout.

## 12.9. Giám sát (Monitoring)

### Các dạng suy giảm

| Dạng | Định nghĩa | Ví dụ | Phát hiện |
|---|---|---|---|
| **Data drift (covariate shift)** | $P(X)$ thay đổi | Chiến dịch thu hút khách trẻ hơn | PSI, KS, Wasserstein, Jensen–Shannon |
| **Prior / label shift** | $P(Y)$ thay đổi | Đối thủ giảm giá, churn tăng | Tỷ lệ dự đoán dương; tỷ lệ nhãn khi về |
| **Concept drift** | $P(Y\mid X)$ thay đổi | Chính sách mới đổi quan hệ giá–churn | Metric hiệu năng khi có nhãn |
| Data quality | Pipeline lỗi | Cột null toàn bộ; đổi đơn vị | Contract, null rate, volume |
| **Training–serving skew** | Feature tính khác nhau | Logic SQL ở serving khác pandas ở training | So phân phối feature log ở serving với training |

### Population Stability Index (PSI) và các khoảng cách phân phối

$$PSI = \sum_{i=1}^{B}(A_i - E_i)\ln\frac{A_i}{E_i}$$

| PSI | Diễn giải (quy ước ngành tín dụng) |
|---|---|
| < 0.1 | Ổn định |
| 0.1 – 0.25 | Thay đổi vừa: theo dõi |
| > 0.25 | Thay đổi lớn: điều tra, cân nhắc retrain |

```python
from churn.monitoring.drift import drift_report, psi

reference = clean_churn(make_churn_data(n=10_000, seed=1))
current = clean_churn(make_churn_data(n=5_000, seed=2))
# Mô phỏng sự cố: (1) chiến dịch thu hút khách trẻ; (2) nguồn dữ liệu đổi đơn vị cước; (3) kênh thanh toán mới
current["age"] = (current["age"] - 8).clip(lower=18)
current.loc[current.sample(frac=0.3, random_state=0).index, "monthly_charges"] *= 1000
current.loc[current.sample(frac=0.2, random_state=1).index, "payment_method"] = "qr-pay"

report = drift_report(reference, current,
                      num_cols=["age", "tenure_months", "monthly_charges", "support_calls", "data_usage_gb"],
                      cat_cols=["contract", "payment_method", "region"])
print(report.round(3).to_string(index=False))

score_psi = psi(served.predict_proba(reference[RAW_FEATURES])[:, 1], served.predict_proba(current[RAW_FEATURES])[:, 1])
print(f"PSI của điểm dự đoán = {score_psi:.3f}")
```

> **Đọc kết quả, một bài học quan trọng.** Input trôi rất mạnh: `monthly_charges` bị nhân 1 000 ở 30% bản ghi, xuất hiện kênh thanh toán mới, khách trẻ hơn. Vậy mà **PSI của điểm dự đoán chỉ ≈ 0.02**. Lý do là `Winsorizer` trong pipeline cắt cước về phân vị 99% của train, còn mức `qr-pay` lạ được xử lý như "infrequent/unknown". Mô hình **bền vững**, nhưng chính sự bền vững đó **che giấu sự cố dữ liệu** (sai đơn vị ở nguồn billing). Vì vậy phải giám sát **cả input lẫn output**: chỉ nhìn phân phối điểm sẽ bỏ sót sự cố này.

> Với dữ liệu lớn, KS test gần như luôn cho p ≈ 0 kể cả khi khác biệt không đáng kể. Hãy dựa vào **độ lớn khác biệt** (PSI, KS statistic, Wasserstein chuẩn hóa) và **ngưỡng nghiệp vụ**, không dựa vào p-value.

### Evidently: báo cáo drift tự động

**[Docs]** Evidently ≥ 0.7 dùng API mới: `Report([...presets]).run(current_data, reference_data)` trả về **Snapshot**, có thể `save_html`, `dict()`/`json()`, hoặc đẩy lên Evidently UI/Cloud. Các preset chính: `DataDriftPreset`, `DataSummaryPreset`, `ClassificationPreset`, `RegressionPreset`. Mặc định Evidently tự chọn phép đo theo kiểu cột và cỡ mẫu: K-S/χ² cho mẫu nhỏ, **Wasserstein (chuẩn hóa)** / **Jensen–Shannon** cho mẫu lớn.

```python
from evidently import Report
from evidently.presets import DataDriftPreset

cols = ["age", "tenure_months", "monthly_charges", "support_calls", "contract", "payment_method"]
snapshot = Report([DataDriftPreset()]).run(current_data=current[cols], reference_data=reference[cols])
snapshot.save_html("drift_report.html")
drift_metrics = [m for m in snapshot.dict()["metrics"] if m["metric_name"].startswith("ValueDrift")]
for m in drift_metrics:
    print(f"{m['metric_name'][:75]:75s} value={float(m['value']):.3f}")
```

### Hiệu năng khi nhãn đến trễ

Nhãn churn chỉ có sau 30–37 ngày. Trong khoảng thời gian đó:

1. Giám sát **proxy**: drift input, drift điểm, tỷ lệ dự đoán dương, calibration theo cohort cũ đã có nhãn.
2. **Ước lượng hiệu năng không cần nhãn**: CBPE (*Confidence-Based Performance Estimation*, thư viện NannyML) dùng xác suất **đã calibrate** để ước lượng confusion matrix kỳ vọng. Phương pháp này chỉ đúng khi không có concept drift.
3. Khi nhãn về: tính metric thật **theo cohort chấm điểm**, so với lúc kiểm định.

```python
def expected_confusion(proba: np.ndarray, threshold: float) -> dict:
    """Ước lượng TP/FP/FN/TN kỳ vọng từ xác suất đã calibrate (ý tưởng của CBPE)."""
    act = proba >= threshold
    return {"E[TP]": float(proba[act].sum()), "E[FP]": float((1 - proba[act]).sum()),
            "E[FN]": float(proba[~act].sum()), "E[TN]": float((1 - proba[~act]).sum())}


p_cur = served.predict_proba(clean_churn(make_churn_data(n=5_000, seed=21))[RAW_FEATURES])[:, 1]
ec = expected_confusion(p_cur, 0.33)
print({k: round(v) for k, v in ec.items()},
      "| precision ước lượng =", round(ec["E[TP]"] / (ec["E[TP]"] + ec["E[FP]"]), 3))
```

### Bảng giám sát đề xuất

| Nhóm | Metric | Ngưỡng cảnh báo ví dụ | Tần suất |
|---|---|---|---|
| Hệ thống | p95/p99 latency, error rate, throughput, CPU/RAM | p95 > 200 ms; lỗi > 1% | Real-time (Prometheus/Grafana) |
| Dữ liệu | Null rate, schema, volume, category mới, freshness | Null tăng > 5 điểm %; volume ±30% | Mỗi batch |
| Drift input | PSI/Wasserstein các feature quan trọng | PSI > 0.25 | Hằng ngày/tuần |
| Drift output | PSI điểm, tỷ lệ dự đoán dương | PSI > 0.1 | Hằng ngày |
| Hiệu năng | PR-AUC, recall, calibration theo cohort khi có nhãn | Giảm > 10% so với kiểm định | Khi nhãn về |
| Kinh doanh | Tỷ lệ giữ chân nhóm được gọi so với holdout | Theo OEC | Hằng tháng |

Mỗi cảnh báo phải có **người nhận (on-call)** và **runbook**: kiểm tra gì, ai quyết định rollback hay retrain.

## 12.10. Retraining và orchestration

| Chiến lược | Mô tả | Phù hợp |
|---|---|---|
| **Định kỳ** | Hằng tuần/tháng | Đơn giản, dễ vận hành. **Mặc định tốt** |
| Theo trigger | Khi drift/hiệu năng vượt ngưỡng | Môi trường biến động |
| Online / incremental | Cập nhật liên tục | Gợi ý, quảng cáo, streaming |

Retrain **không bao giờ deploy mù quáng**: validate dữ liệu → train → đánh giá out-of-time → **gate** so với champion → shadow/canary → promote (có phê duyệt).

```python norun
# flows/retrain.py — Prefect (2.x/3.x): task có retry, flow có lịch
from prefect import flow, get_run_logger, task


@task(retries=3, retry_delay_seconds=60)
def ingest(snapshot: str) -> str:
    return f"data/raw/churn_{snapshot}.parquet"


@task
def validate_data(path: str) -> str:
    from churn.data.validate import validate
    validate(pd.read_parquet(path))
    return path


@task
def train(path: str) -> str:
    from churn.models.train import main
    main("configs/train.yaml")
    return latest_model_version()          # truy vấn registry


@task
def gate(version: str) -> bool:
    from churn.models.registry import promote_if_better
    return promote_if_better("churn-classifier", version)


@flow(name="churn-retrain")
def retrain_flow(snapshot: str = "latest"):
    promoted = gate(train(validate_data(ingest(snapshot))))
    get_run_logger().info("promoted=%s", promoted)


if __name__ == "__main__":
    retrain_flow.serve(name="monthly", cron="0 2 1 * *")
```

```python norun
# dags/churn_retrain.py — Airflow (TaskFlow API)
import pendulum
from airflow.decorators import dag, task


@dag(schedule="0 2 1 * *", start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
     catchup=False, default_args={"retries": 2}, tags=["ml", "churn"])
def churn_retrain():
    @task
    def ingest() -> str: ...

    @task
    def validate(path: str) -> str: ...

    @task
    def train(path: str) -> str: ...

    @task
    def gate(version: str) -> None: ...

    gate(train(validate(ingest())))


churn_retrain()
```

## 12.11. Feature store: chống training–serving skew

**[Docs]** Feast tách **offline store** (dữ liệu lịch sử dùng để tạo tập huấn luyện bằng **point-in-time join**) khỏi **online store** (giá trị mới nhất, độ trễ thấp cho serving). Hai store dùng **cùng một định nghĩa feature**.

```python norun
# feature_repo/features.py — Feast
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

customer = Entity(name="customer", join_keys=["customer_id"])
usage_source = FileSource(path="data/processed/usage_features.parquet", timestamp_field="event_timestamp")
usage_fv = FeatureView(
    name="customer_usage", entities=[customer], ttl=timedelta(days=90),
    schema=[Field(name="support_calls_90d", dtype=Int64), Field(name="avg_bill_6m", dtype=Float32)],
    source=usage_source,
)
# Huấn luyện: store.get_historical_features(entity_df=labels_with_snapshot_ts, features=[...])  -> point-in-time
# Serving   : store.get_online_features(features=[...], entity_rows=[{"customer_id": "C0000001"}])
```

## 12.12. Tối ưu suy luận

| Kỹ thuật | Lợi ích | Công cụ |
|---|---|---|
| Batch inference | Vector hóa, ít overhead | Gom request (micro-batching) |
| Chuyển định dạng | Nhanh 2–10 lần, không phụ thuộc Python | ONNX (`skl2onnx`, `onnxmltools`), Treelite (cây) |
| Lượng tử hóa / distillation | Mô hình nhỏ hơn | ONNX Runtime quantization; train mô hình nhỏ bắt chước mô hình lớn |
| Cache | Bỏ qua tính lại | Redis theo khóa feature hash |
| Tính trước | Độ trễ ≈ 0 | Batch scoring + tra cứu |

## 12.13. Bảo mật, quyền riêng tư, quản trị và chi phí

- **Secret:** biến môi trường hoặc secret manager (Vault, AWS Secrets Manager, GCP Secret Manager). **Không bao giờ** commit vào git.
- **Dữ liệu cá nhân:** giả danh/ẩn danh trước khi vào pipeline (Chương 2.12); phân quyền theo vai trò; tuân thủ **NĐ 13/2023/NĐ-CP**.
- **Lineage:** model version ← run ← data version (DVC hash) ← git commit ← config. MLflow lưu tag `git_commit`, `data_path`.
- **Audit log:** mọi dự đoán quan trọng gồm input hash, output, model version, thời gian.
- **Chuỗi cung ứng:** lock file; `pip-audit`; quét image (`trivy`); chỉ nạp model pickle/cloudpickle từ registry đáng tin. MLflow 3 mặc định dùng **skops** cho scikit-learn chính vì lý do này.
- **Quản trị rủi ro AI:** phân loại rủi ro (EU AI Act: unacceptable / high / limited / minimal), model card, đánh giá fairness định kỳ.
- **Chi phí:** theo dõi chi phí huấn luyện/serving trên mỗi dự đoán; autoscale về mức tối thiểu ngoài giờ.

> **Checklist Chương 12**
> - [ ] Mọi thí nghiệm được log: params, metrics, artifact, data version, git commit.
> - [ ] Registry dùng alias champion/challenger; promote qua quality gate tự động và có phê duyệt.
> - [ ] Có unit, data, behaviour, quality-gate và API tests trong CI; tự chấm theo ML Test Score.
> - [ ] API validate input (pydantic), có `/health` + `/ready`, log có request_id, không lộ stack trace/PII.
> - [ ] Docker image non-root, tag bất biến; Kubernetes có probes, resources, `OMP_NUM_THREADS` khớp `limits.cpu`.
> - [ ] Triển khai qua shadow/canary; rollback = đổi alias; có fallback.
> - [ ] Giám sát hệ thống, dữ liệu, drift, hiệu năng (cả khi nhãn trễ), KPI kinh doanh; mỗi cảnh báo có runbook.
> - [ ] Retraining có lịch/trigger và gate; feature dùng chung định nghĩa giữa training và serving.

### Tài liệu tham khảo Chương 12

- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS.
- Breck, E. et al. (2017). *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction.* IEEE Big Data.
- Google Cloud Architecture Center. *MLOps: Continuous delivery and automation pipelines in machine learning.*
- MLflow 3 documentation: *Tracking*, *LoggedModel*, *Model Registry (aliases)*, *Self-hosting / backend stores*. mlflow.org/docs
- FastAPI documentation: *Concurrency and async/await*, *Lifespan Events*, *Testing*, *Deployment (Docker)*. fastapi.tiangolo.com
- Docker documentation: *Building best practices*. · Kubernetes documentation: *Configure Liveness, Readiness and Startup Probes*; *Horizontal Pod Autoscaling*.
- Evidently documentation (≥ 0.7): *Reports, Presets, Data drift methods*. docs.evidentlyai.com · NannyML documentation: *CBPE*.
- Feast documentation: *Concepts (offline/online store, point-in-time joins)*. docs.feast.dev
- Prefect / Apache Airflow documentation (TaskFlow API).
- Ribeiro, M. T. et al. (2020). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.* ACL.
- Huyen, C. (2022). *Designing Machine Learning Systems*, ch. 7–9. O'Reilly. · Mohandas, G. *Made With ML* (Testing, Serving, CI/CD, Monitoring). · DeepLearning.AI *MLOps Specialization* (Deployment & Monitoring). · Full Stack Deep Learning.
