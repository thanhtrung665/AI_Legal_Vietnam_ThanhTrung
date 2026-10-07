# DATA SCIENCE END-TO-END HANDBOOK

### Cẩm nang Khoa học Dữ liệu chuẩn Production: từ cơ bản đến nâng cao

> **Phiên bản 2.0, tháng 10/2026.** Bản này được viết lại để bám sát **tài liệu chính thức** của các thư viện (scikit-learn User Guide, pandas User Guide, LightGBM/XGBoost/CatBoost docs, Optuna, MLflow, imbalanced-learn, statsmodels, SHAP, Evidently, FastAPI) và giáo trình của các khóa học chuyên sâu (Stanford CS229, Google Machine Learning Crash Course & Rules of ML, DeepLearning.AI MLOps Specialization, Made With ML, Full Stack Deep Learning, Harvard CS109, Kohavi về thử nghiệm có kiểm soát).
>
> **Code đã được chạy kiểm chứng.** Mọi khối code Python không gắn nhãn `norun` đều được script `build/run_snippets.py` thực thi tự động. Package tham chiếu `code/src/churn` có bộ test riêng (`pytest`).

---

## 0.1. Cách dùng handbook

### Cấu trúc một chương

Mỗi chương đi theo cùng một khung:

1. **Mục tiêu & câu hỏi cần trả lời.** Chương này giải quyết vấn đề gì trong dự án thực.
2. **Lý thuyết cốt lõi.** Định nghĩa, công thức, giả định. Đây là phần mà khóa học hàn lâm (CS229, CS109) dạy.
3. **Kỹ thuật & API chính thức.** Tham số quan trọng, hành vi mặc định, những thay đổi giữa các phiên bản, kèm trích dẫn tài liệu gốc.
4. **Code mẫu chạy được.** Viết theo phong cách production: hàm có type hint, pipeline không rò rỉ dữ liệu, seed cố định.
5. **Bẫy thường gặp (Pitfalls).** Lấy từ mục *Common pitfalls* của tài liệu chính thức và kinh nghiệm dự án.
6. **Checklist** và **Tài liệu tham khảo** của chương.

### Quy ước trình bày

| Ký hiệu | Ý nghĩa |
|---|---|
| **[Docs]** | Nội dung tóm lược/diễn giải từ tài liệu chính thức của thư viện. Đường dẫn ở cuối chương |
| **[Course]** | Nội dung tương ứng trong khóa học chuyên sâu |
| **[Kinh nghiệm]** | Bài học thực tế, không phải quy định của thư viện |
| ```` ```python ```` | Khối code **được chạy kiểm chứng** |
| ```` ```python norun ```` | Khối code minh họa cần dịch vụ ngoài (DB, Kafka, API, server) nên không chạy tự động |

### Phiên bản thư viện đã kiểm chứng

| Thư viện | Phiên bản | Thư viện | Phiên bản |
|---|---|---|---|
| Python | 3.13 | scikit-learn | 1.9.1 |
| pandas | 3.0.6 | numpy | 2.x |
| LightGBM | 4.7.0 | XGBoost | 3.4.1 |
| CatBoost | 1.2.10 | Optuna | 5.0.0 |
| MLflow | 3.17.0 | SHAP | 0.52.0 |
| imbalanced-learn | 0.14.2 | statsmodels | 0.15.0 |
| pandera | 0.34.1 | FastAPI | 0.142 |
| fairlearn | 0.14.0 | Evidently | 0.7.x |

> **[Kinh nghiệm]** Thư viện ML thay đổi API nhanh. Ví dụ, trong các phiên bản trên: scikit-learn 1.9 deprecate `TargetEncoder(random_state=...)` (thay bằng truyền splitter vào `cv`). MLflow 3 đổi `artifact_path` thành `name` trong `log_model`, đưa file store `./mlruns` vào chế độ bảo trì và mặc định dùng *skops* để serialize scikit-learn. pandas 3 bật *Copy-on-Write* và kiểu `str` mặc định. Luôn **khóa phiên bản** (lock file) và đọc *release notes* trước khi nâng cấp.

---

## 0.2. Kế hoạch nội dung

| Phần | Chương | Nội dung chính | Nền tảng chính thống |
|---|---|---|---|
| **I. Nền tảng** | 0 | Phương pháp luận, vai trò, cấu trúc repo, môi trường, tái lập | CRISP-DM, Microsoft TDSP, scikit-learn *Controlling randomness* |
| | 1 | Định nghĩa bài toán, metric kinh doanh, thiết kế giải pháp | Google *ML Problem Framing*, *Rules of ML* (Zinkevich) |
| **II. Dữ liệu** | 2 | Thu thập: file, SQL, API, scraping, streaming; data contract; versioning; nhãn | pandas *Scaling to large datasets*, Apache Arrow/Parquet, pandera, DVC |
| | 3 | EDA: hồ sơ dữ liệu, missing, phân phối, quan hệ, leakage, adversarial validation | Tukey (1977), Harvard CS109, scikit-learn |
| | 4 | Thống kê suy luận, A/B test, hồi quy, suy luận nhân quả, survival | statsmodels, SciPy, Kohavi et al. (2020) |
| | 5 | Tiền xử lý & feature engineering, imbalanced data | scikit-learn *Preprocessing*, *Imputation*, *Compose*; imbalanced-learn |
| | 6 | Chuẩn hóa, biến đổi phân phối, giảm chiều | scikit-learn *Preprocessing*, *Decomposition* |
| | 7 | Chia dữ liệu & validation, nested CV, leakage | scikit-learn *Cross-validation*, *Common pitfalls* |
| **III. Mô hình** | 8 | Huấn luyện: lý thuyết, GBM, tuning, ensemble, deep learning, tracking | CS229; LightGBM/XGBoost/CatBoost docs; Optuna; scikit-learn *Ensembles* |
| | 9 | Đánh giá: scoring rules, confusion matrix, threshold, calibration, so sánh thống kê | scikit-learn *Model evaluation*, *Calibration*, *Threshold tuning* |
| | 10 | Giải thích mô hình & fairness | scikit-learn *Inspection*; SHAP; fairlearn; Molnar |
| | 11 | Trực quan hóa & kể chuyện bằng dữ liệu | matplotlib, seaborn, plotly; Cleveland & McGill; Knaflic |
| **IV. Production** | 12 | MLOps: tracking, registry, serving, Docker, CI/CD, monitoring, retraining | Google *MLOps levels*; MLflow; FastAPI; Evidently; Made With ML |
| | 13 | Checklist production, ML Test Score, anti-patterns | Breck et al. (2017); Sculley et al. (2015) |
| **Phụ lục** | A–F | Cheatsheet, chọn metric/thuật toán, thuật ngữ, bản đồ khóa học, tài liệu | |

### Lộ trình đọc đề xuất

| Trình độ | Thứ tự đọc | Trọng tâm |
|---|---|---|
| Mới bắt đầu (0–1 năm) | 0 → 1 → 2 → 3 → 5 → 6 → 7 → 8.1–8.6 → 9.1–9.5 → 11 | Pipeline không leakage, chọn metric đúng |
| Trung cấp (1–3 năm) | Toàn bộ, đặc biệt 4, 7, 8, 9, 10 | Validation đúng kịch bản, tuning, calibration, threshold |
| Senior / ML Engineer | 1, 7, 9, 12, 13 + Phụ lục | Thiết kế hệ thống, kiểm thử ML, giám sát, quản trị |

---

## 0.3. Phương pháp luận dự án

### CRISP-DM (Cross-Industry Standard Process for Data Mining, 1999–2000)

CRISP-DM vẫn là quy trình được dùng nhiều nhất. Nó chia dự án thành 6 pha **có vòng lặp**:

```text
        ┌──────────────────────┐         ┌──────────────────────┐
        │ 1. Business          │◀───────▶│ 2. Data              │
        │    Understanding     │         │    Understanding     │
        └──────────▲───────────┘         └──────────┬───────────┘
                   │                                ▼
        ┌──────────┴───────────┐         ┌──────────────────────┐
        │ 6. Deployment        │         │ 3. Data Preparation  │◀──┐
        └──────────▲───────────┘         └──────────┬───────────┘   │
                   │                                ▼               │
        ┌──────────┴───────────┐         ┌──────────────────────┐   │
        │ 5. Evaluation        │◀────────│ 4. Modeling          │───┘
        └──────────────────────┘         └──────────────────────┘
```

| Pha | Nhiệm vụ chính (theo CRISP-DM 1.0 User Guide) | Sản phẩm bàn giao | Chương |
|---|---|---|---|
| 1. Business Understanding | Xác định mục tiêu kinh doanh, đánh giá tình hình (tài nguyên, rủi ro, chi phí–lợi ích), mục tiêu data mining, kế hoạch dự án | Business objectives, success criteria, project plan | 1 |
| 2. Data Understanding | Thu thập dữ liệu ban đầu, mô tả, khám phá, kiểm tra chất lượng | Data description report, data quality report | 2, 3 |
| 3. Data Preparation | Chọn, làm sạch, xây dựng, tích hợp, định dạng dữ liệu | Dataset + dataset description | 4, 5, 6 |
| 4. Modeling | Chọn kỹ thuật, thiết kế kiểm thử, xây mô hình, đánh giá kỹ thuật | Test design, model, model assessment | 7, 8 |
| 5. Evaluation | Đánh giá theo tiêu chí kinh doanh, rà soát quy trình, quyết định bước tiếp | Approved models, list of next actions | 9, 10 |
| 6. Deployment | Kế hoạch triển khai, giám sát & bảo trì, báo cáo cuối, retrospective | Deployment & monitoring plan, final report | 11, 12 |

**Hạn chế của CRISP-DM** và cách các khung hiện đại bổ sung:

- CRISP-DM ra đời trước thời kỳ MLOps, nên pha *Deployment* rất mỏng: không nói tới giám sát drift, retraining, CI/CD. Chương 12 bổ sung theo mô hình **MLOps maturity** của Google.
- Không mô tả vai trò và cộng tác nhóm. **Microsoft TDSP** (Team Data Science Process) bổ sung vai trò (Group Manager, Team Lead, Project Lead, Individual Contributor), cấu trúc repo chuẩn và các template tài liệu (Project Charter, Data Report, Model Report, Exit Report).
- Không nhấn mạnh thực nghiệm online. Các công ty công nghệ bổ sung vòng **A/B test** sau triển khai (Chương 4.6).

### Vòng đời ML trong production (tổng hợp từ Google, Made With ML, Full Stack Deep Learning)

```text
 Design ──▶ Data ──▶ Model ──▶ Deploy ──▶ Monitor
   ▲          ▲        ▲                     │
   └──────────┴────────┴──── feedback ───────┘
 (scoping, metric)  (label, validate, version)  (train, eval, track)  (serve, test)  (drift, retrain)
```

**[Course]** *Made With ML* (Goku Mohandas) chia khóa MLOps thành: Design → Data → Model → Develop → Utilities → Test → Reproducibility → Production → Data engineering. Khóa *Machine Learning Engineering for Production* (DeepLearning.AI, Andrew Ng) chia theo vòng đời **Scoping → Data → Modeling → Deployment**. Handbook này đi theo đúng mạch đó.

### Tỷ lệ công sức điển hình

| Giai đoạn | % thời gian | Ghi chú |
|---|---|---|
| Hiểu bài toán, thống nhất metric với stakeholder | 10–15% | Sai ở đây thì mọi bước sau đều vô ích |
| Thu thập, làm sạch, gán nhãn | 35–50% | "Data-centric AI" (Andrew Ng): cải thiện dữ liệu thường lợi hơn cải thiện mô hình |
| EDA & feature engineering | 15–20% | |
| Huấn luyện & tuning | 10–15% | Tuning thường chỉ thêm 1–3% |
| Đánh giá, triển khai, giám sát | 15–25% | Bị đánh giá thấp nhất, gây nhiều sự cố nhất |

---

## 0.4. Vai trò trong một đội dữ liệu

| Vai trò | Trách nhiệm chính | Kỹ năng/Công cụ cốt lõi |
|---|---|---|
| Data Analyst | Báo cáo, dashboard, phân tích mô tả & chẩn đoán | SQL, Excel/BI, thống kê mô tả, trực quan hóa |
| Analytics Engineer | Mô hình hóa dữ liệu trong warehouse, chất lượng dữ liệu | SQL, dbt, kiểm thử dữ liệu |
| Data Engineer | Pipeline thu thập, lưu trữ, xử lý dữ liệu lớn | Spark, Kafka, Airflow, cloud |
| **Data Scientist** | Định nghĩa bài toán, EDA, thống kê, mô hình, thực nghiệm | Python, thống kê, ML, giao tiếp kinh doanh |
| ML Engineer | Đưa mô hình vào production, tối ưu suy luận | Software engineering, serving, Docker, K8s |
| MLOps / Platform Engineer | Hạ tầng huấn luyện, registry, CI/CD, giám sát | MLflow, Kubeflow, Terraform, observability |

**[Kinh nghiệm]** Data Scientist "chuẩn production" không cần làm mọi việc của ML Engineer. Nhưng họ phải viết được code **có test, đóng gói được, tái lập được**. Nếu không, chi phí bàn giao sẽ ăn hết giá trị của mô hình.

---

## 0.5. Cấu trúc repository chuẩn

Dựa trên **Cookiecutter Data Science v2** (DrivenData) và cấu trúc package tham chiếu của handbook (`code/`):

```text
churn-prediction/
├── .github/workflows/ci.yml     # lint + test + train + quality gate
├── configs/                     # tham số tách khỏi code (YAML)
│   └── train.yaml
├── data/                        # KHÔNG commit dữ liệu vào git -> DVC / object storage
│   ├── raw/                     # bất biến (immutable), chỉ đọc
│   ├── interim/                 # trung gian
│   ├── processed/               # sẵn sàng huấn luyện
│   └── external/                # dữ liệu bên thứ ba
├── models/                      # artifact cục bộ (production: MLflow Model Registry)
├── notebooks/                   # đặt tên 1.0-tt-eda-churn.ipynb (thứ tự-tác giả-mô tả)
├── reports/figures/
├── src/churn/                   # package Python: mọi logic dùng lại phải ở đây
│   ├── config.py                # pydantic: cấu hình có kiểu, đọc secret từ env
│   ├── data/                    # synthetic.py, validate.py, split.py
│   ├── features/                # clean.py, transformers.py, build.py
│   ├── models/                  # train.py, evaluate.py, registry.py
│   ├── monitoring/drift.py
│   ├── serving/api.py
│   └── viz/style.py
├── tests/                       # unit / data / model / api
├── Dockerfile
├── Makefile
├── pyproject.toml
├── .pre-commit-config.yaml
└── .env.example                 # mẫu biến môi trường (không chứa secret thật)
```

Nguyên tắc của Cookiecutter Data Science:

1. **Dữ liệu thô là bất biến.** Không bao giờ sửa tay file trong `data/raw`. Mọi biến đổi phải là code chạy lại được.
2. **Notebook để khám phá và trao đổi, không để sản xuất.** Logic dùng lại phải chuyển vào `src/` và có test.
3. **Phân tích là một DAG.** Mỗi bước có input/output rõ ràng (Makefile, DVC pipeline, Airflow).
4. **Tách cấu hình và secret khỏi code.** Theo nguyên tắc *Twelve-Factor App* (factor III — Config).

---

## 0.6. Môi trường phát triển

### Quản lý môi trường & phụ thuộc

| Công cụ | Ưu điểm | Khi nào dùng |
|---|---|---|
| **uv** (Astral) | Rất nhanh, quản lý cả Python version, lock file đa nền tảng | Mặc định cho dự án mới |
| pip + venv + pip-tools | Chuẩn, đơn giản | Môi trường hạn chế công cụ |
| Poetry / PDM | Quản lý package + build | Thư viện phát hành lên PyPI |
| conda / mamba | Phụ thuộc C/CUDA/GDAL phức tạp | Deep learning GPU, GIS |
| Docker | Đóng băng toàn bộ hệ điều hành | Production, CI |

```bash
# uv: tạo môi trường, cài package ở chế độ editable, khóa phiên bản
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,extra]"
uv pip compile pyproject.toml -o requirements.lock    # lock file để tái lập
uv pip sync requirements.lock                          # môi trường CI/production
```

### `pyproject.toml` (trích từ package tham chiếu)

```toml
[project]
name = "churn"
version = "2.0.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.0", "pandas>=2.2", "pyarrow>=15", "scipy>=1.13",
    "scikit-learn>=1.6", "lightgbm>=4.3",
    "pandera>=0.24", "pydantic>=2.7", "pydantic-settings>=2.3", "pyyaml>=6.0",
    "mlflow>=3.0", "fastapi>=0.110", "uvicorn[standard]>=0.30",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "httpx>=0.27", "ruff>=0.6", "mypy>=1.10", "pre-commit"]
extra = ["xgboost>=2.1", "catboost>=1.2", "optuna>=4.0", "shap>=0.46", "imbalanced-learn>=0.12",
         "statsmodels>=0.14", "fairlearn>=0.11", "evidently>=0.7", "plotly>=5.22"]

[tool.ruff]
line-length = 110
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "NPY", "PD"]   # PD: pandas-vet, NPY: numpy rules

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

### Chất lượng code tự động: `pre-commit`

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/kynan/nbstripout          # xóa output notebook trước khi commit
    rev: 0.7.1
    hooks:
      - id: nbstripout
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-added-large-files
        args: ["--maxkb=5000"]
      - id: detect-private-key
      - id: check-yaml
```

**[Kinh nghiệm]** Notebook khó review vì diff JSON. Dùng **Jupytext** để ghép mỗi notebook với một file `.py` (định dạng *percent*) và review file `.py` đó. Dùng `nbstripout` để không commit output (output có thể chứa dữ liệu khách hàng).

---

## 0.7. Tái lập kết quả (Reproducibility)

Một kết quả chỉ đáng tin khi tái lập được từ bốn thành phần: **code** (git commit), **dữ liệu** (phiên bản/snapshot), **môi trường** (lock file/Docker image) và **cấu hình + seed**.

### `random_state` trong scikit-learn: điều tài liệu chính thức nhấn mạnh

**[Docs]** Mục *Controlling randomness* trong *Common pitfalls and recommended practices* của scikit-learn:

- Truyền **số nguyên** cho `random_state`: mỗi lần gọi `fit`/`split` cho **cùng kết quả**, vì RNG được khởi tạo lại ở đầu mỗi lần gọi.
- Truyền **`None` hoặc một instance `RandomState`**: mỗi lần gọi cho **kết quả khác nhau**, vì RNG bị "tiêu thụ" và thay đổi trạng thái. Các đối tượng dùng chung một instance sẽ ảnh hưởng lẫn nhau.
- **Khuyến nghị:** với CV splitter, truyền số nguyên để mọi mô hình được so sánh trên **cùng các fold**. Với estimator, khi muốn đánh giá độ bền vững của CV có thể để `None`/instance. Còn khi cần tái lập tuyệt đối giữa các lần chạy thì bỏ mọi `random_state=None`.

```python
import numpy as np
from sklearn.model_selection import KFold

X = np.arange(10).reshape(-1, 1)

# Số nguyên: split() gọi nhiều lần cho cùng fold -> so sánh mô hình công bằng
cv_int = KFold(n_splits=2, shuffle=True, random_state=0)
assert [list(v) for _, v in cv_int.split(X)] == [list(v) for _, v in cv_int.split(X)]

# Instance RandomState: mỗi lần split() cho fold KHÁC -> hai mô hình bị đánh giá trên dữ liệu khác nhau
cv_rng = KFold(n_splits=2, shuffle=True, random_state=np.random.RandomState(0))
first = [list(v) for _, v in cv_rng.split(X)]
second = [list(v) for _, v in cv_rng.split(X)]
print("Fold giống nhau giữa 2 lần split()?", first == second)
```

### Seed cho toàn bộ stack

```python
import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> np.random.Generator:
    """Đặt seed cho các nguồn ngẫu nhiên phổ biến; trả về numpy Generator để dùng tường minh."""
    random.seed(seed)
    np.random.seed(seed)                      # cho thư viện cũ còn dùng global RNG
    os.environ["PYTHONHASHSEED"] = str(seed)  # chỉ có tác dụng nếu đặt TRƯỚC khi khởi động Python
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return np.random.default_rng(seed)        # API khuyến nghị của NumPy (NEP 19)


rng = seed_everything(42)
print(rng.integers(0, 100, 3))
```

| Thư viện | Cách đảm bảo tái lập |
|---|---|
| NumPy | Dùng `np.random.default_rng(seed)` (Generator, NEP 19) thay cho global `np.random.*` |
| scikit-learn | `random_state=int` cho splitter và estimator |
| LightGBM | `seed`/`random_state`; `deterministic=True` + `force_row_wise=True` nếu cần kết quả giống hệt trên CPU; cố định `num_threads` |
| XGBoost | `random_state`; kết quả GPU có thể khác CPU |
| PyTorch | `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, cố định `num_workers` + `worker_init_fn` của DataLoader |
| Optuna | `TPESampler(seed=...)`; tối ưu song song thì không tái lập hoàn toàn |

---

## 0.8. Bộ dữ liệu xuyên suốt handbook

Toàn bộ code dùng bộ dữ liệu **mô phỏng bài toán dự đoán khách hàng rời bỏ (churn)** của một nhà mạng viễn thông. Dữ liệu sinh từ package tham chiếu `churn.data.synthetic`. Vì biết trước **cơ chế sinh dữ liệu thật (ground truth)**, ta kiểm tra được mỗi kỹ thuật có tìm ra sự thật không. Ví dụ: feature nào thực sự quan trọng, leakage làm metric "đẹp ảo" bao nhiêu.

| Cột | Kiểu | Ý nghĩa | Vấn đề được cài sẵn |
|---|---|---|---|
| `customer_id` | str | Mã khách hàng `C0000001` | 100 bản ghi trùng lặp |
| `signup_date` | datetime | Ngày đăng ký (2022-01 → 2024-06) | Dùng cho time split |
| `age` | float | Tuổi | ~5% thiếu |
| `tenure_months` | int | Số tháng gắn bó | Tác động âm lên churn |
| `monthly_charges` | float | Cước tháng | 30 ngoại lai ×8 (lỗi nhập liệu) |
| `total_charges` | float | Tổng cước ≈ monthly × tenure | Đa cộng tuyến |
| `contract` | str | month-to-month / one-year / two-year | 50 bản ghi sai chuẩn hóa `Month-to-Month` |
| `payment_method` | str | e-wallet / bank-transfer / credit-card / cash | ~3% thiếu |
| `region` | str | 30 vùng `R00`…`R29` | Cardinality cao, **không** có tác động thật |
| `support_calls` | int | Số cuộc gọi hỗ trợ (Poisson) | Tác động dương mạnh |
| `data_usage_gb` | float | Dung lượng dữ liệu (log-normal) | Lệch phải; ~8% thiếu; **không** có tác động thật |
| `churn` | int | Nhãn: 1 = rời bỏ | ~16.5% positive (mất cân bằng) |

Cơ chế sinh nhãn (log-odds):

$$\text{logit}\,P(\text{churn}) = -2.2 + 1.3\cdot\mathbb{1}[\text{month-to-month}] - 0.03\cdot\text{tenure} + 0.012\cdot(\text{charges}-70) + 0.35\cdot\text{calls} + 0.4\cdot\mathbb{1}[\text{cash}] - 0.01\cdot(\text{age}-38)$$

```python
from churn.data.synthetic import make_churn_data

df = make_churn_data(n=20_000, seed=42)
print(df.shape)                                   # (20100, 12): 20 000 khách + 100 dòng trùng
print(df["churn"].mean().round(3))                # tỷ lệ churn
print(df.isna().mean().round(3)[lambda s: s > 0]) # các cột có missing
print(df.head(3).T)
```

---

## 0.9. Bản đồ khóa học chuyên sâu

| Khóa học / Giáo trình | Đơn vị | Phù hợp chương | Ghi chú |
|---|---|---|---|
| CS229 *Machine Learning* | Stanford (Andrew Ng, Tengyu Ma…) | 6, 8, 9 | Nền tảng toán: GLM, SVM, bias–variance, learning theory |
| CS109 *Data Science* | Harvard | 2, 3, 4, 11 | Thu thập, EDA, thống kê, trực quan hóa |
| *Machine Learning Specialization* | DeepLearning.AI / Stanford Online | 5–9 | Cập nhật của khóa ML kinh điển |
| *MLOps Specialization* (MLEP) | DeepLearning.AI | 1, 2, 12 | Scoping, data-centric, deployment, monitoring |
| *Machine Learning Crash Course*, *Problem Framing*, *Data Prep*, *Rules of ML* | Google for Developers | 1, 5, 9, 12 | Thực hành, ngắn gọn, nhiều bài học production |
| *Made With ML* | Goku Mohandas (Anyscale) | 0, 7, 12, 13 | MLOps end-to-end, testing, CI/CD |
| *Full Stack Deep Learning* | FSDL | 1, 12 | Dự án ML thực tế, tổ chức đội, hạ tầng |
| *Practical Deep Learning for Coders* | fast.ai | 8 | Deep learning thực hành, tabular & NLP |
| Kaggle Learn (Pandas, Feature Engineering, Intermediate ML, Data Leakage) | Kaggle | 3, 5, 7 | Bài tập ngắn có chấm điểm |
| *Trustworthy Online Controlled Experiments* | Kohavi, Tang, Xu (2020) | 4 | Chuẩn mực A/B test trong ngành |
| *Interpretable Machine Learning* | Christoph Molnar | 10 | Sách mở về XAI |
| *Designing Machine Learning Systems* | Chip Huyen (Stanford CS329S) | 1, 12 | Thiết kế hệ thống ML |

> **Checklist Chương 0**
> - [ ] Repo theo cấu trúc chuẩn; logic dùng lại nằm trong `src/` và có test.
> - [ ] Phụ thuộc được khóa phiên bản; có Dockerfile cho production.
> - [ ] Secret nằm trong biến môi trường hoặc `.env` (đã có trong `.gitignore`).
> - [ ] `pre-commit` chạy ruff, strip notebook output, chặn file lớn và private key.
> - [ ] Seed và `random_state=int` được đặt trong config; CV splitter dùng số nguyên.
> - [ ] Biết dự án đang ở pha nào của CRISP-DM và sản phẩm bàn giao của pha đó.

### Tài liệu tham khảo Chương 0

- Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide.* SPSS.
- Microsoft. *Team Data Science Process (TDSP).* learn.microsoft.com/azure/architecture/data-science-process
- DrivenData. *Cookiecutter Data Science v2.* cookiecutter-data-science.drivendata.org
- scikit-learn User Guide. *Common pitfalls and recommended practices → Controlling randomness.* scikit-learn.org/stable/common_pitfalls.html
- NumPy. *NEP 19 — Random number generator policy.* numpy.org/neps/nep-0019-rng-policy.html
- Wiggins, A. *The Twelve-Factor App.* 12factor.net
- Mohandas, G. *Made With ML — MLOps course.* madewithml.com
