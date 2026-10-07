# DATA SCIENCE END-TO-END HANDBOOK

### Cẩm nang Khoa học Dữ liệu chuẩn Production — từ cơ bản đến nâng cao

> **Phiên bản:** 1.0 — 10/2026
> **Ngôn ngữ code:** Python ≥ 3.10
> **Đối tượng:** Data Analyst, Data Scientist, ML Engineer (Junior → Senior), Tech Lead, sinh viên/nghiên cứu sinh ngành Khoa học dữ liệu.

---

## Lời nói đầu

Sau hơn 15 năm làm dữ liệu — từ các mô hình chấm điểm tín dụng chạy batch hằng đêm, hệ thống gợi ý xử lý hàng chục nghìn request mỗi giây, đến các dự án NLP/LLM gần đây — tôi nhận ra một điều: **mô hình hiếm khi là phần khó nhất**. Phần khó nằm ở chỗ đặt đúng bài toán, có dữ liệu sạch và đáng tin, đánh giá trung thực, và giữ cho mô hình *vẫn đúng* sau 6 tháng chạy production.

Cuốn handbook này được viết theo đúng thứ tự một dự án thực tế diễn ra. Mỗi chương gồm:

1. **Mục tiêu & khái niệm cốt lõi** — hiểu *tại sao* trước khi biết *làm thế nào*.
2. **Kỹ thuật từ cơ bản → nâng cao** — kèm tiêu chí lựa chọn.
3. **Code mẫu** — viết theo phong cách production (có type hint, hàm tái sử dụng, không rò rỉ dữ liệu).
4. **Bẫy thường gặp (Pitfalls)** — những lỗi tôi đã thấy lặp lại ở nhiều team.
5. **Checklist** — để review trước khi chuyển sang bước kế tiếp.

---

## 0.1. Kế hoạch biên soạn handbook

| Phần | Chương | Nội dung chính | Thư viện chủ đạo |
|---|---|---|---|
| **I. Nền tảng** | 0 | Kế hoạch, vòng đời dự án, cấu trúc repo, môi trường | `uv`, `poetry`, `git`, `pre-commit` |
| | 1 | Định nghĩa bài toán & thiết kế giải pháp | — |
| **II. Dữ liệu** | 2 | Thu thập dữ liệu (file, SQL, API, scraping, streaming), data contract, versioning | `pandas`, `polars`, `SQLAlchemy`, `requests`, `httpx`, `pandera`, `DVC` |
| | 3 | Phân tích khám phá dữ liệu (EDA) | `pandas`, `ydata-profiling`, `missingno`, `seaborn` |
| | 4 | Phân tích dữ liệu & thống kê suy luận (kiểm định, A/B test, bootstrap) | `scipy`, `statsmodels`, `pingouin` |
| | 5 | Tiền xử lý & Feature Engineering | `scikit-learn`, `category_encoders`, `imbalanced-learn` |
| | 6 | Chuẩn hóa & biến đổi phân phối | `scikit-learn` |
| | 7 | Chia dữ liệu & chiến lược Validation | `scikit-learn` |
| **III. Mô hình** | 8 | Huấn luyện mô hình, tối ưu siêu tham số, ensemble, deep learning | `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, `Optuna`, `PyTorch` |
| | 9 | Đánh giá mô hình: metrics, confusion matrix, threshold, calibration | `scikit-learn`, `scipy` |
| | 10 | Giải thích mô hình (XAI) & Fairness | `SHAP`, `scikit-learn.inspection`, `fairlearn` |
| | 11 | Trực quan hóa dữ liệu & kết quả | `matplotlib`, `seaborn`, `plotly`, `streamlit` |
| **IV. Production** | 12 | MLOps: tracking, registry, serving, Docker, CI/CD, monitoring, drift, retraining | `MLflow`, `FastAPI`, `Docker`, `GitHub Actions`, `Evidently`, `Prefect/Airflow` |
| | 13 | Checklist Production & Anti-patterns | — |
| **Phụ lục** | A–E | Cheatsheet thư viện, bảng chọn metric, bảng chọn thuật toán, thuật ngữ, tài liệu tham khảo | — |

### Lộ trình đọc đề xuất

- **Người mới (0–1 năm):** Chương 0 → 1 → 2 → 3 → 5 → 6 → 7 → 8 (phần cơ bản) → 9 → 11.
- **Trung cấp (1–3 năm):** Toàn bộ, tập trung chương 4, 7, 8 (nâng cao), 9 (threshold, calibration), 10.
- **Senior / ML Engineer:** Chương 1, 7, 9, 12, 13 — đặc biệt phần leakage, validation, monitoring, retraining.

---

## 0.2. Vòng đời dự án Data Science (CRISP-DM mở rộng cho Production)

```text
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ 1. Business      │──▶│ 2. Data          │──▶│ 3. EDA &         │
│    Understanding │   │    Collection    │   │    Analysis      │
└──────────────────┘   └──────────────────┘   └────────┬─────────┘
         ▲                                             │
         │                                             ▼
┌────────┴─────────┐   ┌──────────────────┐   ┌──────────────────┐
│ 8. Monitoring &  │   │ 5. Modeling &    │◀──│ 4. Preprocessing │
│    Retraining    │   │    Tuning        │   │    & Features    │
└────────▲─────────┘   └────────┬─────────┘   └──────────────────┘
         │                      ▼
┌────────┴─────────┐   ┌──────────────────┐
│ 7. Deployment    │◀──│ 6. Evaluation &  │
│    (Serving)     │   │    Validation    │
└──────────────────┘   └──────────────────┘
```

Nguyên tắc vàng:

1. **Vòng lặp, không phải thác nước.** EDA thường làm thay đổi định nghĩa bài toán; đánh giá thường buộc quay lại thu thập dữ liệu.
2. **Baseline trước, mô hình phức tạp sau.** Một baseline đơn giản trả lời câu hỏi "mô hình có đáng làm không?".
3. **Mọi thứ phải tái lập được (reproducible):** code (git), dữ liệu (DVC/snapshot), môi trường (lock file/Docker), tham số (config), random seed.
4. **Pipeline hóa từ sớm.** Notebook dùng để khám phá; code production nằm trong module có test.

### Tỷ lệ thời gian thực tế trong một dự án

| Giai đoạn | % thời gian (kinh nghiệm thực tế) |
|---|---|
| Hiểu bài toán, làm việc với stakeholder | 10–15% |
| Thu thập, làm sạch, tiền xử lý dữ liệu | 40–50% |
| EDA & feature engineering | 15–20% |
| Huấn luyện & tuning | 10–15% |
| Đánh giá, triển khai, giám sát | 15–20% |

---

## 0.3. Cấu trúc repository chuẩn Production

Dựa trên *Cookiecutter Data Science v2*, được điều chỉnh qua nhiều dự án:

```text
churn-prediction/
├── .github/workflows/          # CI/CD: lint, test, train, deploy
│   └── ci.yml
├── configs/                    # Cấu hình tách khỏi code (YAML)
│   ├── data.yaml
│   ├── train.yaml
│   └── serve.yaml
├── data/                       # KHÔNG commit vào git — quản lý bằng DVC
│   ├── raw/                    # Dữ liệu gốc, bất biến (immutable)
│   ├── interim/                # Dữ liệu trung gian
│   ├── processed/              # Dữ liệu sẵn sàng huấn luyện
│   └── external/               # Dữ liệu bên thứ 3
├── models/                     # Artifact mô hình (hoặc dùng MLflow registry)
├── notebooks/                  # Đặt tên: 01-tt-eda-churn.ipynb (số-tác giả-mô tả)
├── reports/figures/            # Hình ảnh, báo cáo
├── src/churn/                  # Package Python chính
│   ├── __init__.py
│   ├── config.py               # Đọc config (pydantic-settings)
│   ├── data/
│   │   ├── ingest.py           # Thu thập dữ liệu
│   │   ├── validate.py         # Data contract (pandera)
│   │   └── split.py            # Chia dữ liệu
│   ├── features/
│   │   └── build.py            # Feature engineering (sklearn transformers)
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── monitoring/
│   │   └── drift.py
│   └── serving/
│       └── api.py              # FastAPI
├── tests/
│   ├── unit/
│   ├── data/                   # Test chất lượng dữ liệu
│   └── model/                  # Test hành vi mô hình
├── Dockerfile
├── dvc.yaml                    # Định nghĩa pipeline DVC
├── Makefile                    # make data / make train / make test
├── pyproject.toml              # Dependencies + cấu hình ruff, pytest, mypy
├── .pre-commit-config.yaml
├── .env.example                # Mẫu biến môi trường (KHÔNG chứa secret thật)
└── README.md
```

### `pyproject.toml` mẫu

```toml
[project]
name = "churn"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.5",
    "lightgbm>=4.3",
    "optuna>=3.6",
    "pandera>=0.20",
    "mlflow>=2.14",
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "ruff>=0.5", "mypy>=1.10", "pre-commit", "jupyterlab"]
viz = ["matplotlib>=3.8", "seaborn>=0.13", "plotly>=5.22", "shap>=0.45"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "SIM", "PD", "NPY"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --cov=src/churn --cov-report=term-missing"
```

### `Makefile` mẫu

```makefile
.PHONY: install lint test data train serve

install:
	pip install -e ".[dev,viz]" && pre-commit install

lint:
	ruff check src tests && ruff format --check src tests && mypy src

test:
	pytest

data:
	python -m churn.data.ingest --config configs/data.yaml

train:
	python -m churn.models.train --config configs/train.yaml

serve:
	uvicorn churn.serving.api:app --host 0.0.0.0 --port 8000
```

### Quản lý cấu hình với `pydantic-settings`

```python
# src/churn/config.py
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainConfig(BaseModel):
    target: str = "churn"
    test_size: float = Field(0.2, gt=0, lt=1)
    random_state: int = 42
    n_trials: int = 50
    model_params: dict = {}


class Secrets(BaseSettings):
    """Secret đọc từ biến môi trường / file .env — KHÔNG bao giờ hard-code."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    db_url: str = "sqlite:///local.db"
    mlflow_tracking_uri: str = "file:./mlruns"


def load_config(path: str | Path) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        return TrainConfig(**yaml.safe_load(f))
```

---

## 0.4. Bộ dữ liệu xuyên suốt handbook

Để code mẫu chạy được ngay mà không phụ thuộc dữ liệu bên ngoài, toàn bộ handbook dùng một bộ dữ liệu **mô phỏng bài toán dự đoán khách hàng rời bỏ (Customer Churn)** của một công ty viễn thông. Bộ dữ liệu được thiết kế có chủ đích các "vấn đề thực tế": giá trị thiếu, ngoại lai, mất cân bằng lớp, biến phân loại có nhiều mức, biến thời gian.

```python
# src/churn/data/synthetic.py
import numpy as np
import pandas as pd


def make_churn_data(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Sinh dữ liệu churn mô phỏng, có missing, outlier và mất cân bằng lớp (~15-20% churn)."""
    rng = np.random.default_rng(seed)

    signup = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    tenure_months = rng.integers(1, 72, n)
    monthly_charges = rng.normal(70, 25, n).clip(10, 200)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(["e-wallet", "bank-transfer", "credit-card", "cash"], n)
    region = rng.choice([f"R{i:02d}" for i in range(30)], n)          # high-cardinality
    support_calls = rng.poisson(1.5, n)
    data_usage_gb = rng.lognormal(mean=2.5, sigma=0.8, size=n)        # lệch phải
    age = rng.normal(38, 12, n).clip(18, 85)

    # Xác suất churn phụ thuộc phi tuyến vào các biến (ground truth logic)
    logit = (
        -2.2
        + 1.3 * (contract == "month-to-month")
        - 0.03 * tenure_months
        + 0.012 * (monthly_charges - 70)
        + 0.35 * support_calls
        + 0.4 * (payment == "cash")
        - 0.01 * (age - 38)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:07d}" for i in range(n)],
            "signup_date": signup,
            "age": age.round(0),
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": (monthly_charges * tenure_months).round(2),
            "contract": contract,
            "payment_method": payment,
            "region": region,
            "support_calls": support_calls,
            "data_usage_gb": data_usage_gb.round(2),
            "churn": churn.astype(int),
        }
    )

    # Bơm "vấn đề thực tế"
    for col, rate in {"age": 0.05, "data_usage_gb": 0.08, "payment_method": 0.03}.items():
        df.loc[rng.random(n) < rate, col] = np.nan
    outlier_idx = rng.choice(n, size=30, replace=False)
    df.loc[outlier_idx, "monthly_charges"] *= 8                          # ngoại lai do lỗi nhập liệu
    df.loc[rng.choice(n, 50, replace=False), "contract"] = "Month-to-Month"  # lỗi chuẩn hóa chuỗi
    df = pd.concat([df, df.sample(100, random_state=seed)], ignore_index=True)  # bản ghi trùng
    return df


if __name__ == "__main__":
    data = make_churn_data()
    print(data.shape, data["churn"].mean().round(3))
    data.to_parquet("data/raw/churn.parquet", index=False)
```

---

## 0.5. Thiết lập môi trường

```bash
# Cách 1: uv (nhanh nhất, khuyến nghị 2025+)
pip install uv
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,viz]"

# Cách 2: conda / mamba (khi cần thư viện C/CUDA phức tạp)
mamba create -n ds python=3.11 -y && mamba activate ds
pip install -e ".[dev,viz]"

# Khóa phiên bản để tái lập
uv pip compile pyproject.toml -o requirements.lock
```

### `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/kynan/nbstripout       # xoá output notebook trước khi commit
    rev: 0.7.1
    hooks:
      - id: nbstripout
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-added-large-files
        args: ["--maxkb=5000"]
      - id: detect-private-key
```

### Tái lập kết quả (Reproducibility)

```python
import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
```

> **Checklist Chương 0**
> - [ ] Repo có cấu trúc rõ ràng, tách `src/`, `configs/`, `tests/`, `notebooks/`.
> - [ ] Dependencies được khóa phiên bản (lock file).
> - [ ] Secret nằm trong `.env`, `.env` có trong `.gitignore`.
> - [ ] `pre-commit` đã cài: lint, format, strip notebook output.
> - [ ] Có hàm `seed_everything` và seed được ghi vào config.
