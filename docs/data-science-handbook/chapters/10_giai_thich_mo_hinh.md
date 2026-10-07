# CHƯƠNG 10. GIẢI THÍCH MÔ HÌNH (XAI) & FAIRNESS

## 10.1. Vì sao cần giải thích?

- **Tin tưởng & chấp nhận:** CSKH cần biết *vì sao* khách bị đánh dấu nguy cơ cao để tư vấn đúng.
- **Debug:** phát hiện leakage, feature vô lý (mô hình dựa vào `customer_id`?).
- **Tuân thủ pháp lý:** quyết định tín dụng, tuyển dụng, bảo hiểm đòi hỏi giải thích được (EU AI Act, quy định ngân hàng).
- **Insight kinh doanh:** yếu tố nào thúc đẩy churn → hành động phòng ngừa.

| Phạm vi | Câu hỏi | Kỹ thuật |
|---|---|---|
| **Toàn cục (global)** | Feature nào quan trọng nhất với mô hình? | Permutation importance, mean \|SHAP\|, PDP |
| **Cục bộ (local)** | Vì sao khách hàng X bị dự đoán churn 82%? | SHAP waterfall, LIME |
| **Hình dạng quan hệ** | Churn thay đổi thế nào khi tenure tăng? | PDP, ICE, ALE, SHAP dependence |
| **Phản thực tế (counterfactual)** | Cần thay đổi gì để khách không churn? | DiCE |

## 10.2. Feature importance — các loại và cạm bẫy

| Loại | Ưu | Nhược |
|---|---|---|
| Impurity/split-based (`feature_importances_`) | Có sẵn, nhanh | **Thiên vị** biến liên tục và cardinality cao; tính trên train |
| Gain-based (LightGBM `importance_type="gain"`) | Tốt hơn split count | Vẫn tính trên train |
| **Permutation importance** | Model-agnostic, tính trên **validation** | Sai lệch khi các feature tương quan mạnh |
| **SHAP** | Có nền tảng lý thuyết (Shapley), cả global & local | Tốn tính toán (trừ TreeSHAP); giải thích *mô hình*, không phải *nhân quả* |

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# pipe: Pipeline(prep, LGBMClassifier) đã fit; X_valid, y_valid từ các chương trước
result = permutation_importance(
    pipe, X_valid, y_valid, scoring="average_precision",
    n_repeats=10, random_state=0, n_jobs=-1,
)
perm = (pd.DataFrame({"mean": result.importances_mean, "std": result.importances_std},
                     index=X_valid.columns)
          .sort_values("mean", ascending=False))
print(perm.round(4))
# Hoán vị trên CỘT GỐC (trước tiền xử lý) -> importance theo ngôn ngữ nghiệp vụ
```

## 10.3. SHAP (SHapley Additive exPlanations)

Với mỗi dự đoán: $f(x) = \phi_0 + \sum_{j=1}^{M}\phi_j$, trong đó $\phi_0$ là giá trị kỳ vọng (base value) và $\phi_j$ là đóng góp của feature *j*. Với mô hình cây, giá trị SHAP mặc định nằm trên thang **log-odds**.

```python
import shap

prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
X_valid_t = pd.DataFrame(prep.transform(X_valid), columns=prep.get_feature_names_out(),
                         index=X_valid.index)

explainer = shap.TreeExplainer(clf)
sample = X_valid_t.sample(2000, random_state=0)
sv = explainer(sample)                   # shap.Explanation

# 1) Global: beeswarm — tầm quan trọng + chiều tác động
shap.plots.beeswarm(sv, max_display=15)

# 2) Global: bar — mean |SHAP|
shap.plots.bar(sv, max_display=15)

# 3) Dependence: quan hệ phi tuyến + tương tác
shap.plots.scatter(sv[:, "tenure_months"], color=sv[:, "monthly_charges"])

# 4) Local: giải thích 1 khách hàng
i = int(np.argmax(clf.predict_proba(sample)[:, 1]))
shap.plots.waterfall(sv[i], max_display=10)
```

> Nếu `sv.values` có 3 chiều (một số phiên bản trả về cho cả 2 lớp), dùng `sv[..., 1]` để lấy lớp positive.

### Sinh "lý do" dễ hiểu cho người dùng cuối (reason codes)

```python
def top_reasons(sv_row, k: int = 3) -> list[str]:
    contrib = pd.Series(sv_row.values, index=sv_row.feature_names)
    top = contrib[contrib > 0].nlargest(k)
    return [f"{feat} = {sv_row.data[list(sv_row.feature_names).index(feat)]:.2f} "
            f"(+{val:.2f} log-odds)" for feat, val in top.items()]


print(top_reasons(sv[i]))
# ['support_calls = 1.80 (+0.61 log-odds)', 'contract_month-to-month = 1.00 (+0.55 log-odds)', ...]
```

Trong production, ánh xạ tên feature kỹ thuật sang câu tiếng Việt: `"support_calls" → "Gọi tổng đài hỗ trợ nhiều lần"`.

## 10.4. Partial Dependence (PDP), ICE và ALE

```python
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(14, 4))
PartialDependenceDisplay.from_estimator(
    pipe, X_valid.sample(3000, random_state=0),
    features=["tenure_months", "monthly_charges", "support_calls"],
    kind="both",             # "average" = PDP, "individual" = ICE, "both" = cả hai
    subsample=200, centered=True, random_state=0, ax=ax,
)
```

- **PDP** giả định feature độc lập → sai lệch khi feature tương quan (tạo ra tổ hợp phi thực tế như tenure=1 nhưng total_charges rất lớn).
- **ALE** (Accumulated Local Effects — thư viện `alibi`, `PyALE`) khắc phục vấn đề này.
- **ICE** khác nhau nhiều giữa các dòng → có tương tác mạnh.

## 10.5. Mô hình "glass-box" — giải thích được từ bản chất

```python
# Explainable Boosting Machine (Microsoft InterpretML): GAM + boosting, độ chính xác gần GBM
from interpret.glassbox import ExplainableBoostingClassifier

ebm = ExplainableBoostingClassifier(interactions=10, random_state=0)
# ebm.fit(X_train_clean, y_train); ebm.explain_global().visualize()
```

Logistic Regression + WoE (scorecard) vẫn là chuẩn mực trong ngân hàng nhờ tính minh bạch tuyệt đối.

## 10.6. Fairness — công bằng giữa các nhóm

| Tiêu chí | Định nghĩa | Ý nghĩa |
|---|---|---|
| Demographic parity | $P(\hat Y=1 \mid A=a)$ bằng nhau giữa các nhóm | Tỷ lệ được chọn như nhau |
| Equal opportunity | TPR bằng nhau giữa các nhóm | Người "xứng đáng" có cơ hội như nhau |
| Equalized odds | TPR **và** FPR bằng nhau | |
| Predictive parity | Precision bằng nhau | |
| Calibration within groups | Xác suất được calibrate trong từng nhóm | |

> Định lý bất khả thi (Kleinberg et al., 2016; Chouldechova, 2017): khi tỷ lệ cơ sở khác nhau giữa các nhóm, không thể đồng thời thỏa mãn calibration và equalized odds. **Chọn tiêu chí là quyết định nghiệp vụ/đạo đức**, cần stakeholder tham gia.

```python
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                               equalized_odds_difference, selection_rate)
from sklearn.metrics import precision_score, recall_score

threshold = 0.35
y_pred = (pipe.predict_proba(X_valid)[:, 1] >= threshold).astype(int)
age_group = pd.cut(X_valid["age"], bins=[0, 30, 45, 60, 120],
                   labels=["<30", "30-45", "45-60", "60+"]).astype(str)

mf = MetricFrame(
    metrics={"selection_rate": selection_rate, "recall": recall_score,
             "precision": precision_score},
    y_true=y_valid, y_pred=y_pred, sensitive_features=age_group,
)
print(mf.by_group.round(3))
print("Chênh lệch lớn nhất:\n", mf.difference().round(3))
print("Demographic parity diff:",
      round(demographic_parity_difference(y_valid, y_pred, sensitive_features=age_group), 3))
print("Equalized odds diff:",
      round(equalized_odds_difference(y_valid, y_pred, sensitive_features=age_group), 3))
```

Giảm thiểu bias: (1) **Pre-processing** — cân bằng lại dữ liệu, loại proxy feature; (2) **In-processing** — ràng buộc fairness khi huấn luyện (`fairlearn.reductions.ExponentiatedGradient`); (3) **Post-processing** — ngưỡng khác nhau theo nhóm (`fairlearn.postprocessing.ThresholdOptimizer`) — cần cân nhắc pháp lý.

## 10.7. Model Card — tài liệu hóa mô hình

```markdown
# Model Card: churn-classifier v3
- **Mục đích sử dụng:** xếp hạng khách hàng có nguy cơ rời bỏ trong 30 ngày để CSKH liên hệ.
- **Không dùng cho:** quyết định từ chối dịch vụ, định giá cá nhân hóa.
- **Dữ liệu huấn luyện:** snapshot 01/2023–11/2023, 1.2M khách hàng; nhãn = hủy trong 30 ngày.
- **Hiệu năng (test out-of-time 12/2023–02/2024):** PR-AUC 0.61 [0.59–0.63], Recall@Top10% 0.38.
- **Ngưỡng:** 0.35 (tối đa lợi nhuận kỳ vọng).
- **Fairness:** chênh lệch recall giữa các nhóm tuổi ≤ 0.05.
- **Hạn chế:** kém chính xác với khách < 3 tháng (thiếu lịch sử); chưa kiểm định cho khách doanh nghiệp.
- **Giám sát:** PSI hằng tuần; retrain khi PR-AUC giảm > 10% hoặc PSI > 0.25.
- **Chủ sở hữu:** Data Science Team — liên hệ: ds-team@company.vn
```

> **Checklist Chương 10**
> - [ ] Importance tính trên validation (permutation/SHAP), không chỉ split-based trên train.
> - [ ] Hướng tác động của feature hợp lý về nghiệp vụ (sanity check).
> - [ ] Có reason codes cho dự đoán cá nhân nếu người dùng cuối cần.
> - [ ] Đã đo fairness theo các nhóm nhạy cảm liên quan.
> - [ ] Có Model Card ghi rõ mục đích, hạn chế, chủ sở hữu.
