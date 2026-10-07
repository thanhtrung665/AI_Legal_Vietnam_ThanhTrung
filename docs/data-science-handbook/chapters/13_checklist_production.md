# CHƯƠNG 13. CHECKLIST PRODUCTION & ANTI-PATTERNS

## 13.1. Checklist tổng hợp trước khi Go-Live

### A. Bài toán & dữ liệu
- [ ] Mục tiêu kinh doanh, hành động sau dự đoán, KPI online đã được stakeholder ký duyệt.
- [ ] Định nghĩa nhãn có cửa sổ thời gian rõ ràng; feature đảm bảo point-in-time.
- [ ] Data contract + validation tự động tại ingestion; dữ liệu thô được version.
- [ ] PII được ẩn danh; tuân thủ quy định bảo vệ dữ liệu.

### B. Mô hình
- [ ] Vượt baseline nghiệp vụ một cách có ý nghĩa thống kê (CI của chênh lệch không chứa 0).
- [ ] Validation mô phỏng production (out-of-time, group); không có leakage (có test tự động).
- [ ] Ngưỡng được chọn theo chi phí/năng lực vận hành; xác suất đã calibrate nếu dùng trực tiếp.
- [ ] Slice analysis & fairness đã được review.
- [ ] Giải thích mô hình hợp lý về nghiệp vụ; có Model Card.

### C. Kỹ thuật
- [ ] Toàn bộ tiền xử lý + mô hình là 1 artifact; tái lập được từ (code commit, data version, config, seed).
- [ ] Test: unit, data, behavior, quality gate, API — chạy trong CI.
- [ ] API: validate input, xử lý giá trị thiếu/lạ, timeout, health/ready, logging không chứa PII.
- [ ] Load test đạt SLA (p95 latency, throughput).
- [ ] Docker image an toàn, phụ thuộc được quét lỗ hổng.

### D. Vận hành
- [ ] Dashboard giám sát: hệ thống, dữ liệu, drift, hiệu năng, KPI kinh doanh.
- [ ] Cảnh báo có người nhận (on-call) và runbook xử lý.
- [ ] Kế hoạch rollback + fallback đã được diễn tập.
- [ ] Chiến lược retrain và quy trình promote có gate.
- [ ] Thu thập nhãn/feedback để đánh giá và huấn luyện lại.

## 13.2. 20 Anti-patterns tôi gặp nhiều nhất

| # | Anti-pattern | Hậu quả | Cách đúng |
|---|---|---|---|
| 1 | Fit scaler/imputer/encoder trên toàn bộ dữ liệu trước split | Metric ảo | Pipeline + CV |
| 2 | Random split cho dữ liệu có thời gian | Hiệu năng production thấp hơn nhiều | Out-of-time split |
| 3 | SMOTE trước cross-validation | Metric ảo nghiêm trọng | `imblearn.pipeline` |
| 4 | Dùng Accuracy cho dữ liệu mất cân bằng | Mô hình "luôn dự đoán 0" trông tốt | PR-AUC, recall, MCC |
| 5 | Ngưỡng mặc định 0.5 | Sai lệch chi phí kinh doanh | Threshold tuning theo chi phí |
| 6 | Tune trên tập test | Ước lượng lạc quan | Validation riêng / nested CV |
| 7 | Feature có sau sự kiện target | Leakage — mô hình vô dụng khi deploy | Kiểm tra thời điểm sinh feature |
| 8 | Cùng thực thể ở train và test | Mô hình "nhớ" thực thể | Group split |
| 9 | Báo cáo 1 con số không có CI | Quyết định dựa trên nhiễu | Bootstrap CI, mean ± std |
| 10 | Notebook là "production code" | Không tái lập, không test | Module + test + CI |
| 11 | Code feature viết lại khi serving | Training-serving skew | Dùng chung code/pipeline, feature store |
| 12 | Không lưu phiên bản dữ liệu | Không tái lập được kết quả | DVC / snapshot |
| 13 | Xóa ngoại lai trên tập test | Đánh giá không trung thực | Chỉ xử lý theo quy tắc học từ train |
| 14 | Dùng mô hình phức tạp khi chưa có baseline | Không biết giá trị thật của mô hình | Baseline trước |
| 15 | Tin feature importance mặc định của cây | Kết luận sai về yếu tố quan trọng | Permutation importance, SHAP trên validation |
| 16 | Diễn giải SHAP/hệ số như nhân quả | Hành động kinh doanh sai | A/B test, causal inference |
| 17 | Deploy xong là xong | Mô hình suy giảm âm thầm | Monitoring + retrain |
| 18 | Tự động deploy mô hình retrain không có gate | Mô hình tệ lên production | Quality gate + shadow/canary |
| 19 | Hard-code secret, đường dẫn, tham số | Rò rỉ bảo mật, khó thay đổi | `.env`, config YAML |
| 20 | Tối ưu metric offline mà quên KPI online | Mô hình "tốt" nhưng không tạo giá trị | Gắn metric với chi phí, đo bằng A/B test |

## 13.3. Câu hỏi review mô hình (dành cho Tech Lead / Reviewer)

1. Nếu tôi xóa feature quan trọng nhất, hiệu năng giảm bao nhiêu? Feature đó có sẵn tại thời điểm dự đoán không?
2. Phân phối điểm của tập test có giống tập train không (adversarial AUC)?
3. Mô hình hoạt động thế nào với khách hàng mới (cold start), giá trị thiếu, category chưa từng thấy?
4. Cải thiện so với baseline có ý nghĩa thống kê và có giá trị kinh doanh không?
5. Điều gì xảy ra nếu nguồn dữ liệu X ngừng cập nhật 3 ngày?
6. Ai được cảnh báo khi mô hình suy giảm, và họ làm gì?
7. Làm sao rollback trong 5 phút?
