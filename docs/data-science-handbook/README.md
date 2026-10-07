# Data Science End-to-End Handbook (v2.0)

Cẩm nang Khoa học Dữ liệu chuẩn Production, từ định nghĩa bài toán, thu thập dữ liệu, EDA, thống kê suy luận, tiền xử lý, chuẩn hóa, chia dữ liệu, huấn luyện, đánh giá (metrics, confusion matrix, threshold, calibration), giải thích mô hình, trực quan hóa đến MLOps.

**Điểm khác biệt của v2:**

- Nội dung bám sát **tài liệu chính thức** (scikit-learn User Guide, pandas 3, LightGBM/XGBoost tuning guides, Optuna, MLflow 3, imbalanced-learn, statsmodels, SHAP, fairlearn, Evidently, FastAPI) và **khóa học chuyên sâu** (Stanford CS229, Harvard CS109, Google ML guides, DeepLearning.AI MLOps, Made With ML, Kohavi). Mỗi chương có mục *Tài liệu tham khảo*.
- **Mọi khối code đều được chạy kiểm chứng** bằng `build/run_snippets.py` trên phiên bản thư viện mới (scikit-learn 1.9, pandas 3.0, LightGBM 4.7, XGBoost 3.4, MLflow 3.17, Optuna 5.0…), không còn cảnh báo deprecation.
- Có **package tham chiếu chạy được** (`code/`) với test, Dockerfile, CI, MLflow registry, FastAPI.
- Ghi lại các **bẫy thật** phát hiện khi kiểm chứng: feature "ngày tham chiếu cố định", oversubscription luồng CPU trong container, `astype(str)` giữ NaN ở pandas 3, mô hình bền vững che giấu sự cố dữ liệu…

## Sản phẩm

| File / thư mục | Mô tả |
|---|---|
| [`Data_Science_End_to_End_Handbook.md`](Data_Science_End_to_End_Handbook.md) | Bản Markdown đầy đủ (ghép từ các chương) |
| [`Data_Science_End_to_End_Handbook.pdf`](Data_Science_End_to_End_Handbook.pdf) | Bản PDF A4: trang bìa, mục lục, công thức, tô màu code |
| [`chapters/`](chapters/) | Từng chương: chỉnh sửa tại đây |
| [`code/`](code/) | Package `churn` tham chiếu: `src/`, `tests/`, `configs/`, `Dockerfile`, `Makefile`, CI |
| [`build/`](build/) | `run_snippets.py` (kiểm chứng code), `build.sh` (xuất PDF) |

## Mục lục

| # | Chương |
|---|---|
| 0 | [Kế hoạch, phương pháp luận (CRISP-DM/TDSP), môi trường, tái lập, bản đồ khóa học](chapters/00_ke_hoach_va_tong_quan.md) |
| 1 | [Định nghĩa bài toán, hệ thống metric, khung giá trị kỳ vọng, uplift](chapters/01_dinh_nghia_bai_toan.md) |
| 2 | [Thu thập dữ liệu: file/Parquet, pandas 3, Polars/DuckDB, SQL point-in-time, API, streaming, data contract, DVC, nhãn, quyền riêng tư](chapters/02_thu_thap_du_lieu.md) |
| 3 | [EDA: profiling, cơ chế missing, thống kê bền vững, IV/MI, đa cộng tuyến, ngoại lai đa biến, leakage, adversarial validation](chapters/03_eda.md) |
| 4 | [Thống kê suy luận: kiểm định, effect size, đa kiểm định, bootstrap, A/B test (SRM, peeking, CUPED), hồi quy, nhân quả, survival](chapters/04_phan_tich_du_lieu.md) |
| 5 | [Tiền xử lý & feature engineering: imputation, encoding (TargetEncoder), custom transformer, feature selection, imbalanced](chapters/05_tien_xu_ly.md) |
| 6 | [Chuẩn hóa, biến đổi phân phối, biến đổi target (smearing), giảm chiều](chapters/06_chuan_hoa.md) |
| 7 | [Chia dữ liệu & validation: out-of-time, group, purged CV, nested CV, danh mục leakage](chapters/07_chia_du_lieu.md) |
| 8 | [Huấn luyện: ERM & loss nhất quán, GBM chuyên sâu, Optuna, ensemble, chẩn đoán, MLflow, song song hóa](chapters/08_huan_luyen_mo_hinh.md) |
| 9 | [Đánh giá: confusion matrix, ROC/PR/DET, threshold tuning, calibration, metric hồi quy, conformal, so sánh thống kê](chapters/09_danh_gia_mo_hinh.md) |
| 10 | [Giải thích mô hình (permutation, PDP/ICE, SHAP) & fairness (fairlearn)](chapters/10_giai_thich_mo_hinh.md) |
| 11 | [Trực quan hóa & data storytelling](chapters/11_truc_quan_hoa.md) |
| 12 | [MLOps: testing (ML Test Score), MLflow registry, serving, Docker/K8s, CI/CD, deploy, monitoring, retraining](chapters/12_mlops.md) |
| 13 | [Checklist go-live, anti-patterns, review, postmortem, lộ trình 12 tuần](chapters/13_checklist_production.md) |
| — | [Phụ lục: cheatsheet, thay đổi API, chọn metric/thuật toán, thuật ngữ, bản đồ khóa học, tài liệu](chapters/14_phu_luc.md) |

## Chạy và kiểm chứng

```bash
cd docs/data-science-handbook/code
pip install -e ".[dev,extra]"
pytest -q                                                   # test của package tham chiếu
cd ..
python build/run_snippets.py                                # chạy mọi khối code của handbook
./build/build.sh                                            # ghép Markdown + xuất PDF (pandoc + Chromium)
```

Khối code gắn nhãn ```` ```python norun ```` cần dịch vụ ngoài (DB, Kafka, PyTorch, server) nên không chạy tự động.
