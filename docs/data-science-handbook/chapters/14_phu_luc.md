# PHỤ LỤC

## Phụ lục A. Cheatsheet thư viện theo giai đoạn

| Giai đoạn | Thư viện | Dùng để | Tài liệu chính thức |
|---|---|---|---|
| Thao tác dữ liệu | `pandas` (3.x), `polars`, `duckdb`, `pyarrow` | DataFrame, SQL trên file, Arrow/Parquet | pandas.pydata.org · docs.pola.rs · duckdb.org |
| Dữ liệu lớn | `pyspark`, `dask`, `ray` | Xử lý phân tán | spark.apache.org · docs.dask.org |
| Thu thập | `SQLAlchemy` 2.x, `httpx`, `tenacity`, `beautifulsoup4`, `playwright`, `scrapy`, `confluent-kafka` | DB, API, scraping, streaming | docs.sqlalchemy.org · python-httpx.org |
| Chất lượng dữ liệu | `pandera`, `great_expectations`, `soda-core` | Data contract | pandera.readthedocs.io |
| Versioning | `dvc`, Delta Lake, lakeFS | Phiên bản dữ liệu | dvc.org/doc |
| EDA | `ydata-profiling`, `sweetviz`, `missingno` | Báo cáo tự động | |
| Thống kê | `scipy.stats`, `statsmodels`, `pingouin` | Kiểm định, hồi quy, GLM, survival, chuỗi thời gian | docs.scipy.org · statsmodels.org |
| Thực nghiệm & nhân quả | `statsmodels.stats.power`, `DoWhy`, `EconML`, `CausalML` | Power, A/B, uplift | |
| Tiền xử lý | `scikit-learn`, `category_encoders`, `feature-engine`, `imbalanced-learn`, `optbinning` | Pipeline, encoding, resampling | scikit-learn.org · imbalanced-learn.org |
| Văn bản tiếng Việt | `underthesea`, `pyvi`, `transformers` (PhoBERT), `sentence-transformers` | Tách từ, embedding | |
| Mô hình cổ điển | `scikit-learn` | Linear, cây, SVM, KNN, clustering | |
| Gradient boosting | `lightgbm`, `xgboost`, `catboost` | Mô hình bảng mạnh nhất | lightgbm.readthedocs.io · xgboost.readthedocs.io · catboost.ai |
| Deep learning | `pytorch`, `lightning`, `transformers` | NN, NLP, CV | pytorch.org |
| Chuỗi thời gian | `statsforecast`, `sktime`, `darts` | Dự báo | |
| Survival | `lifelines`, `scikit-survival`, `statsmodels.duration` | Thời gian đến sự kiện | |
| Tuning | `optuna`, `ray[tune]` | Tối ưu siêu tham số | optuna.readthedocs.io |
| Giải thích | `shap`, `scikit-learn.inspection`, `interpret` (EBM), `alibi`, `dice-ml` | XAI | shap.readthedocs.io |
| Fairness | `fairlearn`, `aif360` | Đo và giảm bias | fairlearn.org |
| Conformal | `mapie` | Khoảng dự đoán có bảo đảm | |
| Trực quan hóa | `matplotlib`, `seaborn`, `plotly`, `altair` | Biểu đồ | matplotlib.org · seaborn.pydata.org |
| Dashboard | `streamlit`, `dash`, `gradio` | Ứng dụng dữ liệu | docs.streamlit.io |
| Tracking/Registry | `mlflow` (3.x), `wandb` | Thí nghiệm, mô hình | mlflow.org/docs |
| Serving | `fastapi`, `bentoml`, `kserve`, `onnxruntime` | API, tối ưu suy luận | fastapi.tiangolo.com |
| Orchestration | `airflow`, `prefect`, `dagster` | Pipeline | |
| Feature store | `feast` | Offline/online feature | docs.feast.dev |
| Monitoring | `evidently` (0.7+), `nannyml`, `prometheus`, `grafana` | Drift, hiệu năng | docs.evidentlyai.com |
| Chất lượng code | `ruff`, `mypy`, `pytest`, `pre-commit`, `threadpoolctl` | Lint, type, test, kiểm soát luồng | |

## Phụ lục B. Thay đổi API cần biết (đã kiểm chứng khi viết handbook)

| Thư viện | Cũ (đã deprecate / đổi hành vi) | Mới |
|---|---|---|
| scikit-learn 1.8+ | `LogisticRegression(penalty="l1")` | `LogisticRegression(l1_ratio=1, solver="liblinear" \| "saga")`; `C=np.inf` cho không penalty |
| scikit-learn 1.9 | `TargetEncoder(shuffle=True, random_state=0)` | `TargetEncoder(cv=StratifiedKFold(5, shuffle=True, random_state=0))` |
| scikit-learn 1.6+ | `CalibratedClassifierCV(model, cv="prefit")` | `CalibratedClassifierCV(FrozenEstimator(model))` |
| scikit-learn 1.4+ | `mean_squared_error(..., squared=False)` | `root_mean_squared_error(...)` |
| scikit-learn 1.2+ | `OneHotEncoder(sparse=False)` | `OneHotEncoder(sparse_output=False)` |
| scikit-learn 1.9 | (mới) | `metric_at_thresholds`, `confusion_matrix_at_thresholds` |
| pandas 3.0 | Chained assignment `df["a"][m] = v` | `df.loc[m, "a"] = v` (Copy-on-Write luôn bật) |
| pandas 3.0 | Chuỗi kiểu `object` | Kiểu `str` mặc định; datetime không còn mặc định `ns` |
| pandas 3.x → 4.0 | `df.sum(1)` (positional) | `df.sum(axis=1)` |
| MLflow 3 | `log_model(model, artifact_path="model")` | `log_model(sk_model=model, name="model")`; LoggedModel `models:/<model_id>` |
| MLflow 3 | Tracking `./mlruns` (file store) | `sqlite:///mlflow.db` / PostgreSQL |
| MLflow 3 | sklearn serialize mặc định pickle | `skops` (an toàn) ngoài Databricks; custom class cần `skops_trusted_types` hoặc `cloudpickle` + `code_paths` |
| MLflow 2.9+ | Stages `Staging`/`Production` | **Aliases** (`@champion`) |
| SciPy 1.17 | `stats.anderson(x)` | `stats.anderson(x, method="interpolate")` (có `pvalue`) |
| SciPy 1.15+ | `random_state=` trong bootstrap/permutation_test | `rng=` |
| statsmodels 0.15 → 0.16 | `adfuller(x)[1]`, `kpss(x)[1]` (tuple) | `adfuller(x, result_object=True).pvalue` |
| imbalanced-learn 0.14 | `Pipeline` lồng trong `imblearn.pipeline.Pipeline` | Trải phẳng các bước |
| Evidently 0.7 | `from evidently.report import Report`; `report.run(reference_data=..., current_data=...)` | `from evidently import Report`; `Report([...]).run(current_data, reference_data)` trả về Snapshot |
| XGBoost 2+ | `early_stopping_rounds` trong `fit` | Trong constructor |

## Phụ lục C. Bảng chọn metric nhanh

```text
Bài toán?
├── Phân loại
│   ├── Cần xác suất đúng? ─────────────────────▶ Log loss, Brier (+ reliability diagram, ECE)
│   ├── Positive hiếm (<10%)? ──────────────────▶ PR-AUC, Recall@Precision≥x, F-β, MCC
│   ├── Hành động trên top-K? ──────────────────▶ Precision@K, Lift@K, cumulative gain
│   ├── Có ma trận chi phí? ────────────────────▶ Lợi nhuận kỳ vọng (+ threshold tuning)
│   ├── Cân bằng, chi phí lỗi như nhau? ────────▶ Accuracy, F1, ROC-AUC
│   └── Đa lớp ─────────────────────────────────▶ Macro-F1 (lớp hiếm quan trọng) / log loss
├── Hồi quy (theo đại lượng cần dự đoán)
│   ├── Mean ───────────────────────────────────▶ RMSE / R²; Poisson/Gamma/Tweedie deviance cho dữ liệu đếm/dương
│   ├── Median ─────────────────────────────────▶ MAE
│   ├── Quantile / khoảng ──────────────────────▶ Pinball loss + coverage
│   └── Sai số tương đối ───────────────────────▶ WAPE, sMAPE (MAPE chỉ khi y xa 0); MASE cho chuỗi thời gian
├── Xếp hạng / truy hồi ────────────────────────▶ NDCG@K, MAP@K, MRR, Recall@K, F2
└── Phân cụm ───────────────────────────────────▶ Silhouette, Davies–Bouldin (+ ARI/NMI nếu có nhãn) + đánh giá nghiệp vụ
```

## Phụ lục D. Bảng chọn thuật toán

| Tiêu chí | Logistic/Linear | Random Forest | GBM (LGBM/XGB/Cat/HGB) | Neural Net | KNN | SVM |
|---|---|---|---|---|---|---|
| Hiệu năng dữ liệu bảng | Trung bình–Khá (tốt nếu quan hệ gần tuyến tính) | Khá | **Cao nhất** (thường) | Khá–Cao (dữ liệu lớn) | Thấp–TB | Khá |
| Cần scale | Có | Không | Không | Có | Có | Có |
| Xử lý NaN gốc | Không | Có (sklearn ≥ 1.4) | Có | Không | Không | Không |
| Biến phân loại gốc | Không | Không | Có (LGBM, Cat, HGB, XGB) | Embedding | Không | Không |
| Xác suất calibrate sẵn | **Tốt** (khi đúng đặc tả) | Kém (chữ S) | Khá | Khá–kém | Kém | Không có (cần Platt) |
| Giải thích | **Rất tốt** | TB | TB (SHAP) | Kém | TB | Kém |
| Ràng buộc đơn điệu | Dấu hệ số | Không | **Có** | Hạn chế | Không | Không |
| Tốc độ train | Rất nhanh | Nhanh | Nhanh | Chậm | Không cần | Chậm khi n lớn |
| Ngoại suy | Có | **Không** | **Không** | Có | Không | Hạn chế |
| Dữ liệu rất nhỏ | **Tốt** | Tốt | Khá | Kém (trừ TabPFN) | Tốt | Tốt |

## Phụ lục E. Thuật ngữ Anh – Việt

| English | Tiếng Việt |
|---|---|
| Exploratory / Confirmatory Data Analysis | Phân tích khám phá / khẳng định |
| Point-in-time correctness | Đúng thời điểm (không nhìn trộm tương lai) |
| Data leakage / Target leakage | Rò rỉ dữ liệu / rò rỉ nhãn |
| Imputation / Missing indicator | Điền giá trị thiếu / cờ thiếu |
| Cross-fitting | Khớp chéo (mã hóa mỗi fold bằng thống kê các fold khác) |
| Stratified / Group / Out-of-time split | Chia phân tầng / theo nhóm / theo thời gian |
| Nested cross-validation | Kiểm định chéo lồng nhau |
| Early stopping | Dừng sớm |
| Strictly proper scoring rule | Quy tắc chấm điểm thích hợp chặt |
| Calibration / Reliability diagram | Hiệu chuẩn xác suất / biểu đồ độ tin cậy |
| Decision threshold | Ngưỡng quyết định |
| Confusion matrix | Ma trận nhầm lẫn |
| Precision / Recall / Specificity | Độ chính xác dự đoán dương / độ bao phủ (độ nhạy) / độ đặc hiệu |
| Permutation importance | Độ quan trọng hoán vị |
| Partial dependence / ICE | Phụ thuộc riêng phần / kỳ vọng có điều kiện từng mẫu |
| Concept drift / Covariate shift | Trôi khái niệm / dịch chuyển phân phối đầu vào |
| Training–serving skew | Lệch giữa huấn luyện và phục vụ |
| Model registry / Alias | Kho quản lý mô hình / bí danh phiên bản |
| Champion / Challenger | Mô hình đương nhiệm / mô hình thách đấu |
| Shadow / Canary deployment | Triển khai bóng / triển khai thăm dò |
| Oversubscription | Quá tải luồng (nhiều luồng hơn số lõi thật) |
| Overall Evaluation Criterion (OEC) | Tiêu chí đánh giá tổng thể của thí nghiệm |
| Sample Ratio Mismatch (SRM) | Lệch tỷ lệ phân bổ mẫu |

## Phụ lục F. Bản đồ khóa học chuyên sâu theo chương

| Chương | Khóa học / tài liệu chuyên sâu nên học kèm |
|---|---|
| 0–1 | Google *ML Problem Framing* & *Rules of ML*; DeepLearning.AI *MLOps Specialization* (C1: Scoping); FSDL *Lecture: ML Projects*; Chip Huyen *CS329S* |
| 2 | Harvard *CS109* (Data collection & scraping); Kaggle Learn *Pandas*; DeepLearning.AI MLOps (C2: Data); pandas User Guide |
| 3–4 | Harvard *CS109*; Kohavi *Trustworthy Online Controlled Experiments*; Hernán & Robins *Causal Inference: What If*; *Forecasting: Principles and Practice* |
| 5–6 | scikit-learn User Guide & Examples; Kaggle Learn *Feature Engineering*, *Intermediate ML*; Kuhn & Johnson *FES* |
| 7 | scikit-learn *Cross-validation*; Kaggle Learn *Data Leakage*; Andrew Ng *Machine Learning Yearning* |
| 8 | Stanford *CS229*; DeepLearning.AI *Machine Learning Specialization*; fast.ai *Practical Deep Learning*; tài liệu LightGBM/XGBoost/CatBoost/Optuna |
| 9–10 | scikit-learn *Model evaluation*, *Calibration*, *Inspection*; Molnar *Interpretable ML*; fairlearn User Guide |
| 11 | Wilke *Fundamentals of Data Visualization*; Knaflic *Storytelling with Data*; Harvard CS109 (Visualization) |
| 12–13 | *Made With ML* (MLOps); DeepLearning.AI MLOps (C3–C4: Deployment, Monitoring); FSDL; Google *MLOps levels*; MLflow/FastAPI/Evidently docs |

## Phụ lục G. Cách dùng package tham chiếu và kiểm chứng handbook

```bash
cd docs/data-science-handbook/code
pip install -e ".[dev,extra]"
pytest -q                                                   # 18 test: unit, data, behaviour, quality gate, API
python -m churn.models.train --config configs/train.yaml    # huấn luyện + log MLflow (sqlite:///mlflow.db)
make serve                                                  # FastAPI :8000

cd ..
python build/run_snippets.py                                # chạy MỌI khối code của các chương
./build/build.sh                                            # ghép Markdown + xuất PDF
```

## Phụ lục H. Tài liệu tham khảo tổng hợp

**Tài liệu chính thức**

- scikit-learn User Guide (1.9): scikit-learn.org/stable/user_guide.html · Common pitfalls · Examples gallery.
- pandas User Guide (3.0): pandas.pydata.org/docs/user_guide · SciPy reference: docs.scipy.org · statsmodels: statsmodels.org.
- LightGBM: lightgbm.readthedocs.io (Parameters Tuning) · XGBoost: xgboost.readthedocs.io (Notes on Parameter Tuning) · CatBoost: catboost.ai/docs.
- Optuna: optuna.readthedocs.io · MLflow 3: mlflow.org/docs · imbalanced-learn: imbalanced-learn.org · SHAP: shap.readthedocs.io · fairlearn: fairlearn.org · Evidently: docs.evidentlyai.com · pandera: pandera.readthedocs.io · FastAPI: fastapi.tiangolo.com · DVC: dvc.org/doc · Feast: docs.feast.dev.

**Sách**

- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, 3rd ed. O'Reilly.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed. Springer.
- James, G. et al. (2023). *An Introduction to Statistical Learning with Applications in Python.* Springer.
- Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection.* CRC Press.
- Huyen, C. (2022). *Designing Machine Learning Systems.* O'Reilly.
- Kohavi, R., Tang, D. & Xu, Y. (2020). *Trustworthy Online Controlled Experiments.* Cambridge.
- Molnar, C. (2022). *Interpretable Machine Learning*, 2nd ed.
- Provost, F. & Fawcett, T. (2013). *Data Science for Business.* O'Reilly.
- Kleppmann, M. (2017). *Designing Data-Intensive Applications.* O'Reilly.
- Wilke, C. O. (2019). *Fundamentals of Data Visualization.* O'Reilly.

**Bài báo nền tảng**

- Sculley et al. (2015) *Hidden Technical Debt in ML Systems* · Breck et al. (2017) *The ML Test Score* · Kapoor & Narayanan (2023) *Leakage and the reproducibility crisis*.
- Friedman (2001) *Gradient Boosting Machine* · Chen & Guestrin (2016) *XGBoost* · Ke et al. (2017) *LightGBM* · Prokhorenkova et al. (2018) *CatBoost* · Akiba et al. (2019) *Optuna*.
- Gneiting & Raftery (2007) *Strictly Proper Scoring Rules* · Niculescu-Mizil & Caruana (2005) *Predicting Good Probabilities* · Elkan (2001) *Cost-Sensitive Learning*.
- Lundberg & Lee (2017) *SHAP* · Mitchell et al. (2019) *Model Cards* · Grinsztajn et al. (2022) *Tree-based models vs deep learning on tabular data*.

---

*Hết handbook. Phiên bản 2.0, tháng 10/2026. Mọi khối code được kiểm chứng tự động bằng `build/run_snippets.py`. Góp ý qua Pull Request.*
