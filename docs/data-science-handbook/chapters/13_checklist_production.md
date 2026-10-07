# CHƯƠNG 13. CHECKLIST PRODUCTION, ANTI-PATTERNS & QUY TRÌNH REVIEW

**Mục tiêu chương:** cô đọng toàn bộ handbook thành các công cụ dùng hằng ngày: checklist go-live, danh mục anti-pattern (kèm chương tham chiếu), bộ câu hỏi review mô hình, và mẫu postmortem khi sự cố xảy ra.

## 13.1. Checklist trước khi Go-Live

### A. Bài toán & dữ liệu (Chương 1–3)

- [ ] Mục tiêu kinh doanh, hành động sau dự đoán và người chịu trách nhiệm hành động được stakeholder ký duyệt.
- [ ] Đã cân nhắc giải pháp không dùng ML; có baseline heuristic để so sánh.
- [ ] Nhãn có cửa sổ quan sát, gap, label window; loại thực thể không còn hợp lệ tại thời điểm dự đoán.
- [ ] Mọi feature đảm bảo **point-in-time**; không dùng bảng bị ghi đè để tính feature quá khứ.
- [ ] Data contract tự động tại ingestion và trước khi chấm điểm; có kiểm tra volume/freshness.
- [ ] Dữ liệu thô bất biến, có phiên bản (DVC/snapshot); PII được giả danh/ẩn danh theo NĐ 13/2023/NĐ-CP.

### B. Mô hình (Chương 4–10)

- [ ] Validation mô phỏng production (out-of-time / group); có kiểm tra leakage tự động.
- [ ] Mọi bước stateful nằm trong `Pipeline`; target encoding có cross-fitting; resampling nằm trong `imblearn.pipeline`.
- [ ] Vượt baseline **có ý nghĩa thống kê** (CI của chênh lệch không chứa 0); báo cáo test out-of-time có CI.
- [ ] Ngưỡng được chọn theo chi phí/năng lực vận hành trên validation/OOF, đóng gói bằng `FixedThresholdClassifier`.
- [ ] Xác suất đã kiểm tra calibration (calibrate lại nếu dùng trực tiếp).
- [ ] Slice analysis và fairness theo nhóm đã được review; có Model Card.
- [ ] Giải thích mô hình hợp lý về nghiệp vụ (permutation/SHAP trên held-out); behaviour tests đạt.

### C. Kỹ thuật (Chương 0, 8, 12)

- [ ] Huấn luyện là **một lệnh** tái lập được từ (git commit, data version, config, seed); log vào MLflow.
- [ ] Unit, data, behaviour, quality-gate, API tests chạy trong CI; tự chấm **ML Test Score**.
- [ ] API: validate input, xử lý giá trị thiếu/lạ, `/health` + `/ready`, timeout, log có request_id, không lộ PII.
- [ ] Load test đạt SLA (p95/p99) với cấu hình CPU thật; số luồng khớp CPU được cấp.
- [ ] Image non-root, tag bất biến, phụ thuộc đã quét lỗ hổng; secret không nằm trong code/image.
- [ ] Dùng API hiện hành của thư viện (không còn cảnh báo deprecation).

### D. Vận hành (Chương 12)

- [ ] Dashboard giám sát: hệ thống, dữ liệu, drift input/output, hiệu năng theo cohort, KPI kinh doanh.
- [ ] Mỗi cảnh báo có người nhận (on-call) và runbook.
- [ ] Rollback (đổi alias) và fallback (baseline) đã được **diễn tập**.
- [ ] Retraining có lịch/trigger và gate; triển khai qua shadow/canary; có nhóm holdout chống feedback loop.
- [ ] Kế hoạch đo tác động thật (A/B test với OEC và guardrail metrics).

## 13.2. Tự chấm ML Test Score

```python
import pandas as pd

ml_test_score = {
    "Features & Data": {"schema kỳ vọng feature": 1, "mọi feature có ích": 0.5, "chi phí feature được đo": 0,
                        "tuân thủ PII": 1, "code feature có unit test": 1, "pipeline dữ liệu có test": 1,
                        "kiểm tra quyền riêng tư": 0.5},
    "Model Development": {"review & VCS mô hình": 1, "offline ~ online metric": 0, "siêu tham số đã tune": 1,
                          "độ cũ mô hình được biết": 0.5, "so với baseline đơn giản": 1,
                          "chất lượng theo phân khúc": 1, "kiểm tra fairness/inclusion": 0.5},
    "ML Infrastructure": {"huấn luyện tái lập": 1, "mô hình có unit test": 1, "integration test pipeline": 0.5,
                          "kiểm định trước khi serve": 1, "debug từng ví dụ": 0.5, "canary": 0, "rollback": 0.5},
    "Monitoring": {"thay đổi dependency": 0.5, "dữ liệu đầu vào ổn định": 1, "training-serving skew": 0.5,
                   "mô hình không quá cũ": 0.5, "không NaN/Inf output": 1, "hiệu năng tính toán": 1,
                   "chất lượng dự đoán": 0.5},
}
scores = pd.Series({section: sum(items.values()) for section, items in ml_test_score.items()})
print(scores)
print("ML Test Score (min của 4 nhóm) =", scores.min())
# Breck et al. (2017): 0 = nghiên cứu; (0,1] = sơ khai; (1,2] = cơ bản; (2,3] = khá;
# (3,5] = đủ cho hệ thống quan trọng; > 5 = xuất sắc
```

## 13.3. Danh mục anti-pattern

| # | Anti-pattern | Hậu quả | Cách đúng | Chương |
|---|---|---|---|---|
| 1 | Fit scaler/imputer/encoder/feature selection trên toàn bộ dữ liệu | Metric ảo | `Pipeline` + CV | 5, 7 |
| 2 | Random split cho dữ liệu có thời gian | Hiệu năng production thấp hơn nhiều | Out-of-time, `TimeSeriesSplit` | 7 |
| 3 | SMOTE/oversample trước khi chia | Metric ảo nghiêm trọng | `imblearn.pipeline` | 5 |
| 4 | Target encoding bằng `fit(X).transform(X)` | Mô hình tin vào biến nhiễu | `TargetEncoder.fit_transform` | 5 |
| 5 | Accuracy làm metric chính khi mất cân bằng | Mô hình "luôn đoán 0" trông tốt | PR-AUC, recall, MCC, lợi nhuận | 9 |
| 6 | Ngưỡng mặc định 0.5 | Sai lệch chi phí | Threshold tuning theo chi phí | 1, 9 |
| 7 | Tune trên tập test | Ước lượng lạc quan | Validation / nested CV | 7 |
| 8 | Feature sinh ra sau sự kiện target | Leakage, mô hình vô dụng khi deploy | Kiểm tra thời điểm sinh feature | 3, 7 |
| 9 | Cùng thực thể ở train và test | Mô hình "nhớ" thực thể | Group split | 7 |
| 10 | Feature "số ngày kể từ một ngày cố định" | Giá trị lệch/âm trên dữ liệu tương lai | Đo tới thời điểm chấm điểm của từng dòng | 5 |
| 11 | Báo cáo một con số không có CI | Quyết định dựa trên nhiễu | Bootstrap CI, kiểm định ghép cặp | 4, 9 |
| 12 | So sánh mô hình bằng t-test thường trên các fold | p-value quá nhỏ | Corrected resampled t-test | 9 |
| 13 | Tin MDI feature importance | Kết luận sai, thiên vị biến liên tục | Permutation/SHAP trên held-out | 10 |
| 14 | Diễn giải SHAP/hệ số như nhân quả | Hành động kinh doanh sai | A/B test, suy luận nhân quả | 4, 10 |
| 15 | Dùng xác suất sau resampling/class weight như xác suất thật | Lợi nhuận kỳ vọng sai | Calibration | 5, 9 |
| 16 | Biến đổi log target rồi báo cáo tổng/mean | Ước lượng thấp có hệ thống | Smearing; loss Poisson/Gamma | 6 |
| 17 | Notebook là "production code" | Không tái lập, không test | Package + test + CI | 0, 12 |
| 18 | Viết lại logic feature khi serving | Training–serving skew | Dùng chung pipeline; feature store | 12 |
| 19 | Song song lồng nhau (`n_jobs=-1` × mô hình đa luồng) trong container | Chậm hàng chục, hàng trăm lần | Song song một cấp; `OMP_NUM_THREADS` = CPU được cấp | 8, 12 |
| 20 | Deploy xong là xong | Mô hình suy giảm âm thầm | Monitoring + retrain có gate | 12 |
| 21 | Retrain rồi tự động deploy không gate | Mô hình tệ lên production | Quality gate + shadow/canary | 12 |
| 22 | Hard-code secret, đường dẫn, tham số | Rò rỉ bảo mật, khó thay đổi | Env/secret manager, config YAML | 0, 12 |
| 23 | Nạp pickle từ nguồn không tin cậy | Thực thi mã độc | skops / registry kiểm soát | 12 |
| 24 | Bỏ qua cảnh báo deprecation | Pipeline vỡ khi nâng cấp thư viện | Lock phiên bản; chạy test với cảnh báo bật | 0 |
| 25 | Tối ưu metric offline, quên KPI online | Mô hình "tốt" nhưng không tạo giá trị | Hệ thống metric nhất quán; A/B test | 1, 4 |

## 13.4. Bộ câu hỏi review mô hình (dành cho reviewer)

**Bài toán & dữ liệu**

1. Ai dùng dự đoán, dùng lúc nào, hành động gì? Nếu không có mô hình thì hiện tại làm thế nào?
2. Nhãn được định nghĩa thế nào? Có gap không? Thực thể nào bị loại?
3. Feature quan trọng nhất có sẵn **tại thời điểm dự đoán** không? Nếu bỏ nó, hiệu năng giảm bao nhiêu?

**Validation & metric**

4. Cách chia dữ liệu có mô phỏng production không? Adversarial AUC giữa train và test là bao nhiêu?
5. Metric chính có gắn với chi phí kinh doanh không? Ngưỡng được chọn trên dữ liệu nào?
6. Cải thiện so với baseline có CI không? Có ý nghĩa kinh doanh không?

**Rủi ro**

7. Mô hình hoạt động thế nào với khách mới, giá trị thiếu, category chưa thấy, nguồn dữ liệu chậm 3 ngày?
8. Phân khúc nào mô hình kém nhất? Có nhóm nào bị đối xử bất lợi không?
9. Xác suất có được dùng trực tiếp không? Nếu có, đã calibrate chưa?

**Vận hành**

10. Ai nhận cảnh báo khi mô hình suy giảm, và họ làm gì?
11. Rollback trong 5 phút bằng cách nào? Fallback là gì?
12. Khi nào retrain, ai phê duyệt promote?

## 13.5. Mẫu postmortem (blameless) cho sự cố mô hình

```markdown
# Postmortem: [Tên sự cố] — [Ngày]
## Tóm tắt
Một đoạn: chuyện gì xảy ra, ảnh hưởng (khách hàng, doanh thu), thời gian phát hiện và khắc phục.
## Dòng thời gian (giờ Việt Nam)
- 08:00 deploy model v13 (canary 5%)
- 09:30 cảnh báo PSI điểm = 0.41 trên dashboard
- 09:45 on-call xác nhận; 09:50 rollback alias champion -> v12
## Nguyên nhân gốc (5 Whys)
- Vì sao điểm trôi? -> monthly_charges tăng 1000 lần.
- Vì sao? -> nguồn billing đổi đơn vị VNĐ -> đồng mà không thông báo.
- Vì sao không bị chặn? -> data contract chỉ kiểm tra kiểu, không kiểm tra miền giá trị.
## Điều làm tốt / chưa tốt
## Hành động khắc phục (có chủ sở hữu, hạn chót)
- [ ] Thêm kiểm tra miền giá trị và phân phối vào contract (DS, 15/10)
- [ ] Thỏa thuận data contract với team billing (DE, 30/10)
- [ ] Thêm test tích hợp cho thay đổi đơn vị (ML Eng, 20/10)
```

## 13.6. Lộ trình áp dụng handbook trong một dự án 12 tuần

| Tuần | Hoạt động | Sản phẩm | Chương |
|---|---|---|---|
| 1–2 | Framing, metric, design doc; khảo sát nguồn dữ liệu | ML Design Doc ký duyệt | 1 |
| 2–3 | Thu thập, data contract, versioning | Dataset v1 + contract | 2 |
| 3–4 | EDA, phân tích thống kê | Báo cáo EDA + quyết định tiền xử lý | 3, 4 |
| 4–6 | Pipeline tiền xử lý, baseline, mô hình ứng viên | Leaderboard với CI | 5–8 |
| 6–7 | Tuning, threshold, calibration, XAI, fairness | Model Card v1 | 8–10 |
| 7–8 | Package hóa, test, CI, registry | Lệnh train tái lập + CI xanh | 12 |
| 8–9 | Serving (batch/API), Docker, giám sát | Môi trường staging | 12 |
| 9–12 | Shadow → A/B test → rollout; báo cáo kết quả | Quyết định go/no-go dựa trên OEC | 4, 11, 12 |

### Tài liệu tham khảo Chương 13

- Breck, E. et al. (2017). *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction.* IEEE Big Data.
- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS.
- Zinkevich, M. *Rules of Machine Learning.* Google for Developers.
- Google SRE Book: *Postmortem Culture: Learning from Failure.* sre.google
- Kapoor, S. & Narayanan, A. (2023). *Leakage and the reproducibility crisis in ML-based science.* Patterns.
- scikit-learn User Guide: *Common pitfalls and recommended practices*. · imbalanced-learn: *Common pitfalls*.
