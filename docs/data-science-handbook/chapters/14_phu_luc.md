# PHỤ LỤC

## Phụ lục A. Cheatsheet thư viện theo giai đoạn

| Giai đoạn | Thư viện | Dùng để |
|---|---|---|
| Thao tác dữ liệu | `pandas`, `polars`, `duckdb`, `pyarrow` | DataFrame, SQL trên file, Parquet |
| Dữ liệu lớn | `pyspark`, `dask`, `ray` | Xử lý phân tán |
| Thu thập | `SQLAlchemy`, `httpx`, `requests`, `tenacity`, `BeautifulSoup`, `playwright`, `scrapy`, `confluent-kafka` | DB, API, scraping, streaming |
| Chất lượng dữ liệu | `pandera`, `great_expectations`, `soda-core` | Data contract, kiểm tra |
| Versioning | `dvc`, `lakeFS`, Delta Lake | Phiên bản dữ liệu |
| EDA | `ydata-profiling`, `sweetviz`, `missingno`, `dtale` | Báo cáo tự động |
| Thống kê | `scipy.stats`, `statsmodels`, `pingouin` | Kiểm định, hồi quy, chuỗi thời gian |
| Nhân quả | `DoWhy`, `EconML`, `CausalML` | Uplift, causal inference |
| Tiền xử lý | `scikit-learn`, `category_encoders`, `feature-engine`, `imbalanced-learn`, `optbinning` | Pipeline, encoding, resampling |
| Văn bản tiếng Việt | `underthesea`, `pyvi`, `transformers` (PhoBERT), `sentence-transformers` | Tách từ, embedding |
| Mô hình cổ điển | `scikit-learn` | Linear, tree, SVM, KNN, clustering |
| Gradient boosting | `lightgbm`, `xgboost`, `catboost` | Mô hình bảng mạnh nhất |
| Deep learning | `pytorch`, `lightning`, `tensorflow/keras`, `transformers` | NN, NLP, CV |
| Chuỗi thời gian | `statsforecast`, `sktime`, `darts`, `prophet` | Dự báo |
| Tuning | `optuna`, `ray[tune]`, `hyperopt` | Tối ưu siêu tham số |
| Giải thích | `shap`, `lime`, `interpret`, `alibi`, `dice-ml` | XAI |
| Fairness | `fairlearn`, `aif360` | Đo & giảm bias |
| Trực quan hóa | `matplotlib`, `seaborn`, `plotly`, `altair` | Biểu đồ |
| Dashboard | `streamlit`, `dash`, `gradio`, Superset | Ứng dụng dữ liệu |
| Tracking/Registry | `mlflow`, `wandb`, `neptune` | Thí nghiệm, mô hình |
| Serving | `fastapi`, `bentoml`, `kserve`, `triton`, `onnxruntime` | API, tối ưu suy luận |
| Orchestration | `airflow`, `prefect`, `dagster`, `kubeflow` | Pipeline |
| Feature store | `feast` | Offline/online feature |
| Monitoring | `evidently`, `nannyml`, `prometheus`, `grafana` | Drift, hiệu năng |
| Chất lượng code | `ruff`, `mypy`, `pytest`, `pre-commit` | Lint, type, test |

## Phụ lục B. Bảng chọn metric nhanh

```text
Bài toán?
├── Phân loại
│   ├── Cần xác suất chính xác? ──────────────▶ Log loss, Brier, ECE
│   ├── Lớp positive hiếm (<10%)? ────────────▶ PR-AUC, Recall@Precision≥x, F-β, MCC
│   ├── Hành động trên top-K? ────────────────▶ Precision@K, Lift@K, Gain
│   ├── Lớp cân bằng, chi phí lỗi như nhau? ──▶ Accuracy, F1, ROC-AUC
│   └── Đa lớp ───────────────────────────────▶ Macro-F1 (lớp hiếm quan trọng) / Weighted-F1
├── Hồi quy
│   ├── Ngoại lai nhiều / cần dễ hiểu? ───────▶ MAE, MedAE
│   ├── Phạt nặng lỗi lớn? ───────────────────▶ RMSE
│   ├── Sai số tương đối, y > 0? ─────────────▶ RMSLE, MAPE (y xa 0), sMAPE, WAPE
│   └── Dự báo khoảng / phân vị? ─────────────▶ Pinball loss, coverage
├── Xếp hạng / truy hồi ──────────────────────▶ NDCG@K, MAP@K, MRR, Recall@K, F2
└── Phân cụm ─────────────────────────────────▶ Silhouette, Davies–Bouldin (+ ARI/NMI nếu có nhãn)
```

## Phụ lục C. Bảng chọn thuật toán

| Tiêu chí | Logistic/Linear | Random Forest | GBM (LGBM/XGB/Cat) | Neural Net | KNN | SVM |
|---|---|---|---|---|---|---|
| Hiệu năng trên dữ liệu bảng | Trung bình | Khá | **Cao nhất** | Khá–Cao (dữ liệu lớn) | Thấp–TB | Khá |
| Cần scale | Có | Không | Không | Có | Có | Có |
| Xử lý NaN gốc | Không | Có (sklearn ≥ 1.4) | Có | Không | Không | Không |
| Giải thích | **Rất tốt** | TB | TB (SHAP) | Kém | TB | Kém |
| Tốc độ train | Rất nhanh | Nhanh | Nhanh | Chậm | Không cần train | Chậm (n lớn) |
| Tốc độ dự đoán | Rất nhanh | TB | Nhanh | Nhanh (GPU) | Chậm | TB |
| Dữ liệu nhỏ | **Tốt** | Tốt | Khá | Kém | Tốt | Tốt |
| Ngoại suy | Có | **Không** | **Không** | Có | Không | Hạn chế |

## Phụ lục D. Thuật ngữ Anh – Việt

| English | Tiếng Việt |
|---|---|
| Exploratory Data Analysis (EDA) | Phân tích khám phá dữ liệu |
| Feature engineering | Kỹ thuật tạo đặc trưng |
| Data leakage | Rò rỉ dữ liệu |
| Imputation | Điền giá trị thiếu |
| Encoding | Mã hóa |
| Scaling / Normalization / Standardization | Co giãn / Chuẩn hóa min-max / Chuẩn hóa z-score |
| Cross-validation | Kiểm định chéo |
| Hyperparameter tuning | Tối ưu siêu tham số |
| Overfitting / Underfitting | Quá khớp / Chưa khớp |
| Confusion matrix | Ma trận nhầm lẫn |
| Precision / Recall | Độ chính xác (dự đoán dương) / Độ bao phủ (độ nhạy) |
| Calibration | Hiệu chuẩn xác suất |
| Threshold | Ngưỡng quyết định |
| Drift | Trôi dạt (thay đổi phân phối) |
| Model registry | Kho quản lý mô hình |
| Serving / Inference | Phục vụ mô hình / Suy luận |
| Champion / Challenger | Mô hình đương nhiệm / Mô hình thách đấu |
| Point-in-time correctness | Đúng thời điểm (không nhìn trộm tương lai) |
| Training-serving skew | Lệch giữa huấn luyện và phục vụ |

## Phụ lục E. Tài liệu tham khảo

1. Géron, A. — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, 3rd ed., O'Reilly, 2022.
2. Hastie, T., Tibshirani, R., Friedman, J. — *The Elements of Statistical Learning*, 2nd ed., Springer, 2009.
3. Kuhn, M., Johnson, K. — *Feature Engineering and Selection*, CRC Press, 2019.
4. Huyen, C. — *Designing Machine Learning Systems*, O'Reilly, 2022.
5. Molnar, C. — *Interpretable Machine Learning*, 2nd ed., 2022 (christophm.github.io/interpretable-ml-book).
6. Kohavi, R., Tang, D., Xu, Y. — *Trustworthy Online Controlled Experiments*, Cambridge, 2020.
7. López de Prado, M. — *Advances in Financial Machine Learning*, Wiley, 2018.
8. Sculley, D. et al. — *Hidden Technical Debt in Machine Learning Systems*, NeurIPS 2015.
9. Breck, E. et al. — *The ML Test Score: A Rubric for ML Production Readiness*, IEEE Big Data 2017.
10. Grinsztajn, L., Oyallon, E., Varoquaux, G. — *Why do tree-based models still outperform deep learning on tabular data?*, NeurIPS 2022.
11. Ribeiro, M. T. et al. — *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*, ACL 2020.
12. Tài liệu chính thức: scikit-learn.org, lightgbm.readthedocs.io, optuna.org, mlflow.org, fastapi.tiangolo.com, docs.evidentlyai.com.

---

*Hết handbook. Phiên bản 1.0 — 10/2026. Góp ý & cập nhật qua Pull Request.*
