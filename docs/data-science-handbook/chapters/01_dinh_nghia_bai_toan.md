# CHƯƠNG 1. ĐỊNH NGHĨA BÀI TOÁN & THIẾT KẾ GIẢI PHÁP

> *"Rule #1: Don't be afraid to launch a product without machine learning."* — Martin Zinkevich, *Rules of Machine Learning* (Google).

**Mục tiêu chương:** biến một mong muốn kinh doanh mơ hồ ("giảm khách rời bỏ") thành một bài toán ML **đo được, khả thi, không rò rỉ dữ liệu**, có metric gắn với tiền và một baseline để so sánh.

## 1.1. Có nên dùng Machine Learning không?

**[Course]** Khóa *Introduction to Machine Learning Problem Framing* của Google đề xuất trả lời theo thứ tự:

1. **Mục tiêu (goal) là gì?** Phát biểu bằng ngôn ngữ kinh doanh, không nhắc tới mô hình.
2. **Kết quả lý tưởng (ideal outcome)?** Sản phẩm/quy trình sẽ thay đổi thế nào nếu có dự đoán tốt.
3. **Có giải pháp không dùng ML không?** Heuristic, quy tắc, truy vấn SQL. Theo *Rules of ML* (#1, #3), heuristic là baseline bắt buộc. Chỉ dùng ML khi heuristic trở nên quá phức tạp để bảo trì.
4. **Dữ liệu có đủ không?** Có nhãn không, có đủ ví dụ positive không, feature có sẵn **tại thời điểm dự đoán** không.
5. **Dự đoán có hành động được không (actionable)?** Ai dùng, lúc nào, làm gì khác đi.

| Dấu hiệu nên dùng ML | Dấu hiệu chưa nên dùng ML |
|---|---|
| Quy luật phức tạp, nhiều tương tác, khó viết thành quy tắc | Quy tắc đơn giản đã đủ tốt |
| Dữ liệu lớn, có nhãn, lặp lại thường xuyên | Ít dữ liệu, nhãn không đáng tin |
| Chấp nhận được sai số, có cơ chế sửa sai | Sai một lần là thảm họa, không giải thích được |
| Môi trường thay đổi, cần cập nhật liên tục | Quyết định hiếm, một lần |

## 1.2. Khung chuyển đổi bài toán

| Bước | Câu hỏi | Ví dụ Churn |
|---|---|---|
| 1. Mục tiêu kinh doanh | KPI nào cần cải thiện? | Giảm tỷ lệ rời bỏ tháng từ 2.1% xuống 1.8% trong 2 quý |
| 2. Hành động | Dự đoán xong ai làm gì? | CSKH gọi + tặng voucher cho top-K khách nguy cơ cao |
| 3. Đơn vị dự đoán | Dự đoán cho ai, khi nào, bao lâu một lần? | Mỗi thuê bao đang hoạt động, ngày 1 hằng tháng (batch) |
| 4. Nhãn | Định nghĩa chính xác positive? | Hủy dịch vụ trong 30 ngày, bắt đầu sau 7 ngày kể từ ngày chấm điểm |
| 5. Dạng bài toán | Phân loại / hồi quy / xếp hạng / uplift? | Phân loại nhị phân để xếp hạng top-K, tiến tới uplift |
| 6. Ràng buộc | Độ trễ, năng lực vận hành, giải thích, pháp lý | Batch; CSKH gọi tối đa 5 000 khách/tháng; phải có lý do cho từng khách |
| 7. Metric offline | Metric nào phản ánh mục tiêu? | PR-AUC, Precision@5000, lợi nhuận kỳ vọng |
| 8. Metric online | Đo hiệu quả thật thế nào? | Tỷ lệ churn nhóm được gọi so với nhóm đối chứng (A/B test) |
| 9. Baseline | Hiện tại làm thế nào? | Quy tắc: hợp đồng tháng **và** ≥ 3 cuộc gọi khiếu nại |

### Phân loại bài toán và điểm khởi đầu

| Dạng bài toán | Đầu ra | Ví dụ | Khởi đầu (baseline → mạnh) | Loss/metric chuẩn |
|---|---|---|---|---|
| Phân loại nhị phân | Xác suất ∈ [0, 1] | Churn, gian lận | Logistic Regression → GBM | Log loss; PR-AUC |
| Phân loại đa lớp | Phân phối trên K lớp | Phân loại ticket | Multinomial LR → GBM/Transformer | Log loss; macro-F1 |
| Đa nhãn | Tập con nhãn | Gắn tag văn bản | One-vs-Rest; BCE | Hamming loss; micro-F1 |
| Hồi quy | Số thực (mean) | Doanh thu khách hàng | Ridge → GBM | MSE (mean), MAE (median) |
| Hồi quy phân vị | Phân vị | Dự trữ tồn kho P90 | GBM `objective="quantile"` | Pinball loss |
| Dữ liệu đếm | Số nguyên ≥ 0 | Số cuộc gọi | Poisson GLM → GBM Poisson | Poisson deviance |
| Xếp hạng | Thứ tự | Tìm kiếm, gợi ý | BM25 → LambdaMART | NDCG@K, MAP |
| Chuỗi thời gian | Giá trị tương lai | Dự báo nhu cầu | Seasonal naive → ETS/GBM | MASE, WAPE |
| Thời gian đến sự kiện | Hàm sống còn | Khi nào khách rời bỏ | Kaplan–Meier → Cox | C-index |
| Uplift / nhân quả | Hiệu ứng can thiệp | Ai nên nhận voucher | T-/X-learner, Causal Forest | Qini, AUUC |
| Phân cụm | Nhóm | Phân khúc khách hàng | K-Means → GMM/HDBSCAN | Silhouette + đánh giá nghiệp vụ |
| Phát hiện bất thường | Điểm bất thường | Giám sát giao dịch | Isolation Forest | Precision@K trên nhãn kiểm tra |

> **[Kinh nghiệm]** Churn thường được đặt thành "phân loại nhị phân", nhưng câu hỏi kinh doanh thật là: *nếu gọi điện, khách nào sẽ **đổi ý**?* Đó là bài toán **uplift** (mục 1.7). Khách chắc chắn đi dù có gọi ("lost causes") và khách chắc chắn ở lại ("sure things") đều không đáng tốn voucher.

## 1.3. Định nghĩa nhãn và cửa sổ thời gian

Đây là nơi **leakage** sinh ra nhiều nhất. Phải thiết kế ngay từ đầu.

```text
           Observation window (feature)          Gap       Label window
   ├────────────────────────────────────────┤├────────┤├─────────────────────┤
 T − 180 ngày                              T       T+7          T+37
   chỉ dùng dữ liệu có timestamp < T          (độ trễ     churn trong khoảng này
   (point-in-time correctness)                vận hành)   ⇒ label = 1
```

- **Feature** chỉ tính từ dữ liệu có **timestamp < T** và phải *đã có sẵn* trong hệ thống vào lúc T. Dữ liệu đến trễ (late-arriving data) phải được tính theo thời điểm nó *thực sự* có mặt.
- **Gap** mô phỏng độ trễ vận hành: ETL chạy xong, CSKH cần thời gian liên hệ.
- **Label window** dài bao nhiêu là quyết định nghiệp vụ. Ngắn quá thì ít positive, dài quá thì hành động không kịp.
- Khách **đã rời bỏ trước T** phải bị loại khỏi tập chấm điểm, nếu không mô hình học "dự đoán quá khứ".

```python
import numpy as np
import pandas as pd


def make_event_log(n_customers: int = 2_000, seed: int = 0) -> pd.DataFrame:
    """Nhật ký sự kiện mô phỏng: 'activate' khi đăng ký, 'cancel' khi rời bỏ (nếu có)."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n_customers), unit="D")
    life = pd.to_timedelta(rng.exponential(240, n_customers).astype(int), unit="D")
    ids = [f"C{i:07d}" for i in range(n_customers)]
    activate = pd.DataFrame({"customer_id": ids, "event_date": start, "event_type": "activate"})
    cancel = pd.DataFrame({"customer_id": ids, "event_date": start + life, "event_type": "cancel"})
    cancel = cancel[cancel["event_date"] < pd.Timestamp("2025-06-30")]
    return pd.concat([activate, cancel], ignore_index=True)


def build_labels(events: pd.DataFrame, snapshot: str, horizon_days: int = 30,
                 gap_days: int = 7) -> pd.DataFrame:
    """Nhãn point-in-time: khách đang hoạt động tại T, label = hủy trong [T+gap, T+gap+horizon)."""
    t = pd.Timestamp(snapshot)
    start, end = t + pd.Timedelta(days=gap_days), t + pd.Timedelta(days=gap_days + horizon_days)
    activated = set(events.loc[(events.event_type == "activate") & (events.event_date < t), "customer_id"])
    cancelled_before = set(events.loc[(events.event_type == "cancel") & (events.event_date < t), "customer_id"])
    cancel_dates = events.loc[events.event_type == "cancel"].set_index("customer_id")["event_date"]

    active = sorted(activated - cancelled_before)          # loại khách đã rời bỏ trước T
    labels = pd.DataFrame({"customer_id": active, "snapshot_date": t})
    cd = labels["customer_id"].map(cancel_dates)
    labels["churn"] = ((cd >= start) & (cd < end)).astype(int)
    # Khách hủy trong gap: không thể can thiệp kịp -> loại khỏi huấn luyện để nhãn sạch
    in_gap = (cd >= t) & (cd < start)
    return labels[~in_gap].reset_index(drop=True)


events = make_event_log()
for snap in ["2024-06-01", "2024-09-01", "2024-12-01"]:
    lab = build_labels(events, snap)
    print(snap, "n_active =", len(lab), "churn_rate =", round(lab["churn"].mean(), 4))
```

Huấn luyện trên **nhiều snapshot** (ví dụ mỗi tháng một snapshot) làm tăng dữ liệu và giúp mô hình bền với mùa vụ. Khi đó **một khách hàng xuất hiện nhiều lần**, nên bắt buộc phải chia theo nhóm và theo thời gian (Chương 7).

## 1.4. Hệ thống metric: từ KPI kinh doanh đến hàm loss

```text
North-star KPI (doanh thu giữ lại / tháng)
   └── Online metric (tỷ lệ churn nhóm được can thiệp vs đối chứng)          ← A/B test
         └── Offline business metric (lợi nhuận kỳ vọng, Precision@K)        ← chọn ngưỡng/K
               └── Offline ML metric (PR-AUC, log loss)                       ← chọn mô hình
                     └── Training loss (binary cross-entropy)                  ← tối ưu tham số
```

- **Tính nhất quán theo tầng:** mỗi tầng dưới phải là *proxy* tốt của tầng trên. **Định luật Goodhart**: khi một chỉ số trở thành mục tiêu, nó không còn là chỉ số tốt. Tối ưu PR-AUC quá mức mà quên lợi nhuận là một ví dụ.
- **[Course]** Andrew Ng (*Structuring ML Projects*): chọn **một metric tối ưu (optimizing)** để xếp hạng mô hình, và các **metric thỏa mãn (satisficing)** chỉ cần vượt ngưỡng. Ví dụ: tối đa PR-AUC với điều kiện độ trễ < 100 ms, recall nhóm khách VIP ≥ 0.7, chênh lệch recall giữa nhóm tuổi ≤ 0.05.
- **Guardrail metrics** (Kohavi): các chỉ số *không được xấu đi* khi triển khai, như doanh thu/khách, khiếu nại, tỷ lệ hủy dịch vụ do bị làm phiền.

## 1.5. Khung giá trị kỳ vọng (Expected Value framework)

**[Course/Sách]** Provost & Fawcett, *Data Science for Business* (ch. 7) đề xuất đánh giá mô hình bằng **giá trị kỳ vọng**, kết hợp ma trận nhầm lẫn với ma trận lợi ích–chi phí:

$$\mathbb{E}[\text{profit}] = \sum_{(a,\,y)} P(\hat{y}=a,\,y)\cdot b(a, y)$$

| | Thực tế: churn (y=1) | Thực tế: ở lại (y=0) |
|---|---|---|
| **Gọi + voucher** (ŷ=1) | b(1,1) = v·r − c = 500k·0.3 − 50k = **+100k** | b(1,0) = −c = **−50k** |
| **Không làm gì** (ŷ=0) | b(0,1) = 0 *(mất khách là hiện trạng)* | b(0,0) = 0 |

với v = giá trị vòng đời khách (500k), r = xác suất giữ chân thành công khi được can thiệp (30%), c = chi phí can thiệp (50k).

**Ngưỡng tối ưu.** Gọi khách khi lợi ích kỳ vọng dương:

$$p\cdot(v r - c) + (1-p)\cdot(-c) > 0 \iff p > \frac{c}{v\,r} = \frac{50}{150} \approx 0.33$$

Ngưỡng này chỉ đúng khi xác suất **đã được hiệu chuẩn (calibrated)**, xem Chương 9. Ngưỡng mặc định 0.5 làm mất lợi nhuận.

```python
import numpy as np


def expected_profit_per_customer(p: np.ndarray, value: float = 500, retain_rate: float = 0.3,
                                 cost: float = 50) -> np.ndarray:
    """Lợi nhuận kỳ vọng (nghìn VNĐ) nếu can thiệp khách có xác suất churn p (đã calibrate)."""
    return p * (value * retain_rate - cost) + (1 - p) * (-cost)


p = np.array([0.1, 0.2, 0.33, 0.5, 0.8])
print({float(k): float(v) for k, v in zip(p, expected_profit_per_customer(p).round(1))})
print("Ngưỡng hòa vốn p* =", round(50 / (500 * 0.3), 3))
```

## 1.6. Baseline, mức hiệu năng con người và tính khả thi

### Các loại baseline

| Baseline | Ví dụ | Mục đích |
|---|---|---|
| Ngây thơ (naive) | Dự đoán tỷ lệ churn trung bình; `DummyClassifier` | Sàn tối thiểu; PR-AUC của nó bằng **prevalence** |
| Heuristic nghiệp vụ | Quy tắc CSKH đang dùng | Mô hình phải vượt cái **đang có** |
| Mô hình đơn giản | Logistic Regression vài feature | Đo giá trị gia tăng của độ phức tạp |
| Hiệu năng con người | Chuyên gia CSKH đoán | Ước lượng *Bayes error* (Andrew Ng) |

**[Course]** Andrew Ng phân tích sai số thành **avoidable bias** (sai số train so với mức con người/Bayes) và **variance** (chênh lệch train–dev). Biết thành phần nào lớn hơn sẽ quyết định nên làm gì tiếp: mô hình lớn hơn hay thêm dữ liệu/regularization (Chương 8.1).

### Cỡ mẫu tối thiểu

- Với mô hình hồi quy logistic, tài liệu thống kê y sinh dùng quy tắc **EPV (events per variable) ≥ 10–20**: mỗi tham số cần ít nhất 10–20 sự kiện positive (Peduzzi et al., 1996; Riley et al., 2019 đề xuất cách tính cỡ mẫu chính xác hơn).
- Với ML linh hoạt (GBM, NN) cần nhiều hơn đáng kể. Cách thực tế là vẽ **learning curve** (Chương 8) để xem thêm dữ liệu có còn giúp không.

## 1.7. Đặt bài toán dưới dạng uplift (nâng cao)

Gọi T ∈ {0, 1} là can thiệp (gọi điện), Y là churn. Thứ ta cần là **hiệu ứng can thiệp có điều kiện (CATE)**:

$$\tau(x) = \mathbb{E}[Y \mid T=1, X=x] - \mathbb{E}[Y \mid T=0, X=x]$$

Chỉ gọi những khách có τ(x) âm nhiều nhất (giảm churn nhiều nhất). Muốn ước lượng τ(x) **phải có dữ liệu thực nghiệm ngẫu nhiên** (A/B test) hoặc dùng các phương pháp nhân quả với giả định mạnh (Chương 4.8). Dưới đây là **T-learner**: hai mô hình riêng cho nhóm treatment và control.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(1)
n = 20_000
X = pd.DataFrame({"tenure": rng.integers(1, 72, n), "calls": rng.poisson(1.5, n),
                  "m2m": rng.integers(0, 2, n)})
T = rng.integers(0, 2, n)                                  # can thiệp NGẪU NHIÊN (A/B test)
base = -2 + 1.2 * X.m2m - 0.03 * X.tenure + 0.3 * X.calls
true_tau_logit = -1.0 * X.m2m * (X.calls >= 2)             # chỉ nhóm m2m & nhiều khiếu nại được lợi
Y = (rng.random(n) < 1 / (1 + np.exp(-(base + T * true_tau_logit)))).astype(int)

X_tr, X_te, T_tr, T_te, Y_tr, Y_te = train_test_split(X, T, Y, test_size=0.3, random_state=0)
m1 = HistGradientBoostingClassifier(random_state=0).fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
m0 = HistGradientBoostingClassifier(random_state=0).fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
tau_hat = m1.predict_proba(X_te)[:, 1] - m0.predict_proba(X_te)[:, 1]

seg = (X_te.m2m == 1) & (X_te.calls >= 2)
print("τ̂ trung bình nhóm được lợi :", tau_hat[seg].mean().round(3))
print("τ̂ trung bình nhóm còn lại   :", tau_hat[~seg].mean().round(3))
```

Thư viện chuyên dụng: **CausalML** (Uber), **EconML** (Microsoft), **scikit-uplift**. Đánh giá bằng **Qini curve / AUUC** trên dữ liệu thực nghiệm.

## 1.8. Tài liệu thiết kế (ML Design Doc) & quản trị rủi ro

```markdown
# [Tên dự án] — ML Design Doc (v0.3)
## 1. Bối cảnh & mục tiêu: KPI, giá trị kỳ vọng (tiền), người hưởng lợi
## 2. Phi mục tiêu (non-goals): những gì dự án KHÔNG làm
## 3. Định nghĩa bài toán: đơn vị, nhãn, cửa sổ thời gian, dạng bài toán
## 4. Dữ liệu: nguồn, chủ sở hữu, độ trễ, khối lượng, PII, rủi ro chất lượng
## 5. Baseline hiện tại & tiêu chí thành công (offline: optimizing + satisficing; online: A/B)
## 6. Phương pháp: feature, mô hình ứng viên, chiến lược validation (out-of-time?)
## 7. Triển khai: batch/online, SLA độ trễ, tần suất retrain, fallback khi lỗi
## 8. Giám sát: drift, hiệu năng khi có nhãn trễ, KPI kinh doanh, người trực (on-call)
## 9. Rủi ro & đạo đức: fairness, giải thích được, tuân thủ (NĐ 13/2023/NĐ-CP, EU AI Act)
## 10. Kế hoạch & mốc: MVP -> shadow -> A/B -> rollout
## 11. Câu hỏi mở / quyết định cần stakeholder
```

| Rủi ro | Ví dụ | Giảm thiểu |
|---|---|---|
| Dữ liệu | Nhãn churn ghi nhận trễ 2 tuần | Gap trong định nghĩa nhãn; kiểm tra độ trễ |
| Mô hình | Overfit giai đoạn khuyến mãi | Out-of-time validation nhiều giai đoạn |
| Vận hành | CSKH không gọi theo danh sách | Theo dõi tỷ lệ thực thi; đào tạo |
| Đạo đức/pháp lý | Phân biệt đối xử theo tuổi | Đo fairness; loại biến nhạy cảm và biến proxy |
| Phản hồi (feedback loop) | Khách được gọi thì không churn ⇒ nhãn sau này lệch | Giữ nhóm đối chứng ngẫu nhiên (holdout) cố định |

> **Checklist Chương 1**
> - [ ] Có mục tiêu kinh doanh đo được, hành động cụ thể sau dự đoán và người chịu trách nhiệm hành động.
> - [ ] Đã cân nhắc giải pháp không dùng ML; có baseline heuristic.
> - [ ] Nhãn có cửa sổ quan sát, gap, label window; loại khách đã rời bỏ trước T.
> - [ ] Hệ thống metric nhất quán từ KPI đến loss; có optimizing, satisficing và guardrail.
> - [ ] Ma trận lợi ích–chi phí được stakeholder xác nhận; biết ngưỡng hòa vốn.
> - [ ] Có kế hoạch nhóm đối chứng để đo tác động thật và tránh feedback loop.

### Tài liệu tham khảo Chương 1

- Google for Developers. *Introduction to Machine Learning Problem Framing.* developers.google.com/machine-learning/problem-framing
- Zinkevich, M. *Rules of Machine Learning: Best Practices for ML Engineering.* developers.google.com/machine-learning/guides/rules-of-ml
- Provost, F. & Fawcett, T. (2013). *Data Science for Business*, ch. 7 "Decision Analytic Thinking". O'Reilly.
- Ng, A. *Structuring Machine Learning Projects* (DeepLearning.AI) và *Machine Learning Yearning*.
- Huyen, C. (2022). *Designing Machine Learning Systems*, ch. 2. O'Reilly.
- Gutierrez, P. & Gérardy, J.-Y. (2017). *Causal Inference and Uplift Modelling: A Review of the Literature.* PMLR 67.
- Riley, R. D. et al. (2019). *Minimum sample size for developing a multivariable prediction model.* Statistics in Medicine 38(7).
