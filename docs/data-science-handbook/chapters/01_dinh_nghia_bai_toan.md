# CHƯƠNG 1. ĐỊNH NGHĨA BÀI TOÁN & THIẾT KẾ GIẢI PHÁP

> *"Một mô hình hoàn hảo cho một bài toán sai vẫn là một thất bại."*

## 1.1. Từ câu hỏi kinh doanh đến bài toán ML

| Bước | Câu hỏi cần trả lời | Ví dụ (Churn) |
|---|---|---|
| 1. Mục tiêu kinh doanh | Doanh nghiệp muốn cải thiện chỉ số nào? | Giảm tỷ lệ rời bỏ 2 điểm % trong 2 quý |
| 2. Hành động (Action) | Dự đoán xong thì ai làm gì? | CSKH gọi điện + tặng voucher cho top-K khách nguy cơ cao |
| 3. Đơn vị dự đoán | Dự đoán cho đối tượng nào, lúc nào? | Mỗi khách hàng, vào ngày 1 hằng tháng |
| 4. Nhãn (Label) | Định nghĩa chính xác "positive"? | Hủy dịch vụ trong **30 ngày tới** kể từ ngày dự đoán |
| 5. Loại bài toán | Phân loại/hồi quy/xếp hạng/phân cụm/chuỗi thời gian? | Phân loại nhị phân + xếp hạng top-K |
| 6. Ràng buộc | Độ trễ, chi phí, giải thích, pháp lý? | Batch hằng tháng; phải giải thích lý do cho CSKH |
| 7. Metric offline | Metric nào phản ánh mục tiêu? | PR-AUC, Recall@Top-10%, Lift |
| 8. Metric online | Đo hiệu quả thực tế ra sao? | Tỷ lệ giữ chân nhóm treatment vs control (A/B test) |
| 9. Baseline | Hiện tại làm thế nào? | Quy tắc: hợp đồng tháng + >3 cuộc gọi khiếu nại |

### Các dạng bài toán và thuật toán điển hình

| Dạng bài toán | Đầu ra | Ví dụ | Thuật toán khởi đầu |
|---|---|---|---|
| Phân loại nhị phân | 0/1 + xác suất | Churn, gian lận, spam | Logistic Regression → LightGBM |
| Phân loại đa lớp | 1 trong K lớp | Phân loại ticket, ảnh | Multinomial LR → GBM / CNN / Transformer |
| Đa nhãn (multi-label) | Tập con các nhãn | Gắn tag văn bản | One-vs-Rest, BCE loss |
| Hồi quy | Số thực | Giá nhà, doanh thu | Ridge → GBM |
| Xếp hạng (ranking) | Thứ tự | Tìm kiếm, gợi ý | LambdaMART (LightGBM ranker) |
| Chuỗi thời gian | Giá trị tương lai | Dự báo nhu cầu | Seasonal naive → ETS/ARIMA → GBM với lag features |
| Phân cụm | Nhóm | Phân khúc khách hàng | K-Means, HDBSCAN, GMM |
| Phát hiện bất thường | Điểm bất thường | Giám sát hệ thống | Isolation Forest, Autoencoder |
| Suy luận nhân quả | Hiệu ứng can thiệp (uplift) | Ai nên nhận voucher? | T-learner, X-learner, Causal Forest |

> **Lưu ý của chuyên gia:** Bài toán churn ở trên thực chất là bài toán **uplift** — ta không muốn tìm người *sẽ rời bỏ*, mà tìm người *sẽ ở lại nếu được can thiệp*. Hãy bắt đầu bằng mô hình churn, nhưng đề xuất thiết kế A/B test để tiến tới uplift modeling.

## 1.2. Định nghĩa nhãn và cửa sổ thời gian (tránh leakage ngay từ thiết kế)

```text
         Observation window            Gap      Prediction (label) window
   ├───────────────────────────────┤├───────┤├──────────────────────────┤
 T-180d                           T     T+7d                        T+37d
   (chỉ dùng dữ liệu trong khoảng này     (khoảng đệm cho      (churn xảy ra ở đây
    để tạo feature)                        vận hành)             thì label = 1)
```

- **Feature** chỉ được tính từ dữ liệu có sẵn **trước thời điểm T** (point-in-time correctness).
- **Gap** mô phỏng độ trễ vận hành (dữ liệu đến chậm, thời gian CSKH hành động).
- **Label** tính trong cửa sổ tương lai.

```python
import pandas as pd


def build_label(events: pd.DataFrame, snapshot_date: str, horizon_days: int = 30,
                gap_days: int = 7) -> pd.DataFrame:
    """Gán nhãn churn cho mỗi khách hàng tại snapshot_date (point-in-time)."""
    t = pd.Timestamp(snapshot_date)
    start, end = t + pd.Timedelta(days=gap_days), t + pd.Timedelta(days=gap_days + horizon_days)
    active = events.loc[events["event_date"] < t, "customer_id"].unique()
    churned = events.loc[
        (events["event_type"] == "cancel") & events["event_date"].between(start, end),
        "customer_id",
    ].unique()
    labels = pd.DataFrame({"customer_id": active, "snapshot_date": t})
    labels["churn"] = labels["customer_id"].isin(churned).astype(int)
    return labels
```

## 1.3. Từ chi phí kinh doanh đến metric

Xây dựng **ma trận chi phí/lợi ích** cùng stakeholder:

| | Dự đoán: Churn | Dự đoán: Không churn |
|---|---|---|
| **Thực tế: Churn** | TP: giữ được khách (+ 500k – 50k voucher) | FN: mất khách (– 500k) |
| **Thực tế: Không churn** | FP: tốn voucher (– 50k) | TN: 0 |

Từ đây suy ra: FN đắt gấp 10 lần FP → ưu tiên **Recall**, chọn ngưỡng theo **lợi nhuận kỳ vọng** (chương 9), không phải mặc định 0.5.

## 1.4. Tài liệu thiết kế (Design Doc) một trang

```markdown
# [Tên dự án] — ML Design Doc
## 1. Bối cảnh & mục tiêu kinh doanh (KPI, giá trị kỳ vọng)
## 2. Định nghĩa bài toán (đơn vị, nhãn, cửa sổ thời gian, loại bài toán)
## 3. Dữ liệu (nguồn, khối lượng, độ trễ, chủ sở hữu, PII)
## 4. Baseline hiện tại & tiêu chí thành công (offline + online)
## 5. Phương pháp tiếp cận (feature, mô hình, validation)
## 6. Triển khai (batch/online, SLA độ trễ, tần suất retrain)
## 7. Giám sát & rủi ro (drift, fairness, fallback)
## 8. Kế hoạch & mốc thời gian
```

> **Checklist Chương 1**
> - [ ] Có mục tiêu kinh doanh đo được và hành động cụ thể sau dự đoán.
> - [ ] Nhãn định nghĩa rõ ràng, có cửa sổ thời gian và gap.
> - [ ] Có baseline (quy tắc / heuristic) để so sánh.
> - [ ] Metric offline gắn với chi phí kinh doanh; có kế hoạch đo online.
> - [ ] Ràng buộc (độ trễ, giải thích, pháp lý, PII) được ghi nhận.
