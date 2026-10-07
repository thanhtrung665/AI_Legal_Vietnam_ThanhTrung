# CHƯƠNG 9. ĐÁNH GIÁ MÔ HÌNH: METRICS, CONFUSION MATRIX, THRESHOLD & CALIBRATION

**Mục tiêu chương:** đo hiệu năng **đúng thứ cần đo**, **đúng cách** và **trung thực về độ bất định**. Tách bạch hai bài toán mà tài liệu scikit-learn nhấn mạnh: **dự đoán** (ước lượng xác suất tốt) và **ra quyết định** (chọn ngưỡng để hành động).

## 9.1. Khung lý thuyết: dự đoán và ra quyết định

**[Docs]** scikit-learn, *Tuning the decision threshold for class prediction*: bài toán phân loại nên chia làm hai phần:

1. **Bài toán thống kê:** học mô hình ước lượng tốt $P(y \mid X)$. Kết quả đến từ `predict_proba`/`decision_function`. Ví dụ: "khả năng mưa ngày mai là bao nhiêu?"
2. **Bài toán quyết định:** dựa vào xác suất đó để hành động. Kết quả đến từ `predict`, mặc định với ngưỡng cứng 0.5. Ví dụ: "có nên mang ô không?"

Ngưỡng 0.5 "hầu như chắc chắn không lý tưởng cho đa số trường hợp sử dụng". Ví dụ của tài liệu: phát hiện khối u thì bác sĩ ưu tiên recall, nên ngưỡng phải thấp hơn nhiều.

**[Docs]** *Which scoring function should I use?*: đánh giá **chất lượng xác suất** bằng **strictly proper scoring rule** (log loss, Brier score). Hai thước đo này đo cả **calibration** lẫn **discrimination (resolution)**. Còn **chất lượng quyết định** được đo bằng các metric từ confusion matrix tại ngưỡng đã chọn, hoặc bằng lợi nhuận/chi phí.

### Bảng tra metric theo bài toán

| Bài toán | Metric chính | Bổ sung |
|---|---|---|
| Phân loại, cần xác suất đúng | **Log loss, Brier** | ECE, reliability diagram |
| Phân loại mất cân bằng, xếp hạng | **PR-AUC (AP)**, ROC-AUC | Recall@Precision, Precision@K |
| Quyết định nhị phân | Lợi nhuận kỳ vọng, F-β, MCC | Confusion matrix tại ngưỡng |
| Tín dụng | Gini = 2·AUC − 1, KS | PSI (ổn định) |
| Hồi quy (mean) | RMSE, MAE | R², D², MAPE/sMAPE/WAPE |
| Hồi quy phân vị / khoảng | Pinball loss | Coverage, độ rộng khoảng |
| Truy hồi / gợi ý | NDCG@K, MAP@K, MRR, Recall@K | F2 (ưu tiên recall) |
| Phân cụm | Silhouette, Davies–Bouldin, Calinski–Harabasz | ARI/NMI khi có nhãn |

### Thiết lập: hai mô hình để so sánh suốt chương

```python
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score, fbeta_score, log_loss,
                             matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from churn.data.split import time_split
from churn.data.synthetic import make_churn_data
from churn.features.build import RAW_FEATURES, build_preprocessor
from churn.features.clean import clean_churn
from churn.models.evaluate import binary_report, expected_calibration_error, lift_table, probabilistic_report

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 20)
df = clean_churn(make_churn_data(n=30_000, seed=42))
train_df, valid_df, test_df = time_split(df, "signup_date", "2023-10-01", "2024-01-01")
X_train, y_train = train_df[RAW_FEATURES], train_df["churn"]
X_valid, y_valid = valid_df[RAW_FEATURES], valid_df["churn"]
X_test, y_test = test_df[RAW_FEATURES], test_df["churn"]

logit = make_pipeline(build_preprocessor(scale=True), LogisticRegression(max_iter=3000)).fit(X_train, y_train)
# n_jobs=1: mô hình sẽ được fit lại song song trong CV ở các mục sau -> tránh oversubscription (Chương 8.11)
gbm = make_pipeline(build_preprocessor(scale=False), lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, n_jobs=1, verbose=-1, random_state=0)).fit(X_train, y_train)
p_logit, p_gbm = logit.predict_proba(X_valid)[:, 1], gbm.predict_proba(X_valid)[:, 1]
print(pd.DataFrame({"logit": probabilistic_report(y_valid, p_logit),
                    "lightgbm": probabilistic_report(y_valid, p_gbm)}).round(4))
```

## 9.2. Confusion matrix: nền tảng của mọi metric phân loại

```text
                         DỰ ĐOÁN
                    Positive (1)          Negative (0)
            ┌──────────────────────┬──────────────────────┐
 THỰC  P(1) │ TP — bắt đúng churn  │ FN — bỏ sót churn    │ ← sai lầm loại II
 TẾ         ├──────────────────────┼──────────────────────┤
       N(0) │ FP — báo nhầm        │ TN — đúng: ở lại     │
            └──────────────────────┴──────────────────────┘
                  ↑ sai lầm loại I
```

> **Quy ước của scikit-learn:** `confusion_matrix(y_true, y_pred)` trả về **hàng = thực tế, cột = dự đoán**, nhãn theo thứ tự tăng dần `[0, 1]`, tức là `[[TN, FP], [FN, TP]]`. Thứ tự này **ngược** với hình trên. Dùng `.ravel()` để lấy `tn, fp, fn, tp`.

### Các chỉ số dẫn xuất

| Chỉ số | Công thức | Câu hỏi trả lời | Tên khác |
|---|---|---|---|
| **Accuracy** | $\dfrac{TP+TN}{N}$ | Tỷ lệ đúng tổng thể | Vô dụng khi mất cân bằng |
| **Precision** | $\dfrac{TP}{TP+FP}$ | Trong các dự đoán positive, bao nhiêu đúng? | PPV |
| **Recall** | $\dfrac{TP}{TP+FN}$ | Trong các positive thật, bắt được bao nhiêu? | Sensitivity, TPR |
| **Specificity** | $\dfrac{TN}{TN+FP}$ | Trong các negative thật, nhận đúng bao nhiêu? | TNR |
| FPR | $\dfrac{FP}{FP+TN}$ | Tỷ lệ báo động nhầm | Fall-out, 1 − Specificity |
| FNR | $\dfrac{FN}{FN+TP}$ | Tỷ lệ bỏ sót | Miss rate |
| NPV | $\dfrac{TN}{TN+FN}$ | Dự đoán negative đáng tin đến đâu? | |
| **F1** | $\dfrac{2PR}{P+R}$ | Trung bình điều hòa P và R | |
| **F-β** | $\dfrac{(1+\beta^2)PR}{\beta^2P+R}$ | β = 2: recall nặng gấp 4 lần precision | |
| **Balanced accuracy** | $\dfrac{TPR+TNR}{2}$ | Accuracy công bằng giữa các lớp | |
| **MCC** | $\dfrac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | Tương quan dự đoán–thực tế ∈ [−1, 1] | Phi coefficient |
| **Cohen's κ** | $\dfrac{p_o-p_e}{1-p_e}$ | Đồng thuận vượt mức ngẫu nhiên | |
| Youden's J | TPR − FPR | Chọn ngưỡng trên ROC | Informedness |
| Markedness | PPV + NPV − 1 | | |

**Phụ thuộc prevalence:** Precision, NPV, accuracy và F1 **thay đổi khi tỷ lệ positive thay đổi**, kể cả khi mô hình không đổi. TPR, FPR và ROC-AUC thì **không** phụ thuộc prevalence. Hệ quả: precision đo trên tập cân bằng giả tạo (sau resample) không phản ánh precision trong production.

$$\text{Precision} = \frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi + \text{FPR}\cdot(1-\pi)},\qquad \pi = \text{prevalence}$$

### Ví dụ tính tay

Tập kiểm tra 1 000 khách, 200 churn. Mô hình dự đoán 250 churn, trong đó 150 đúng.

| | Dự đoán 1 | Dự đoán 0 | Tổng |
|---|---|---|---|
| Thực tế 1 | TP = 150 | FN = 50 | 200 |
| Thực tế 0 | FP = 100 | TN = 700 | 800 |

- Accuracy = 850/1000 = **0.85**; Precision = 150/250 = **0.60**; Recall = 150/200 = **0.75**; Specificity = 700/800 = **0.875**
- F1 = 2·0.6·0.75/1.35 = **0.667**; F2 = 5·0.6·0.75/(4·0.6 + 0.75) = **0.714**
- MCC = (150·700 − 100·50)/√(250·200·800·750) = 100 000/173 205 = **0.577**
- Mô hình "luôn đoán 0": Accuracy = **0.80**, Recall = 0, MCC = 0. Accuracy gây hiểu lầm.

```python
y_true_toy = np.r_[np.ones(200), np.zeros(800)].astype(int)
y_pred_toy = np.r_[np.ones(150), np.zeros(50), np.ones(100), np.zeros(700)].astype(int)
print(binary_report(y_true_toy, y_pred_toy).round(3).to_dict())
```

### Confusion matrix của mô hình thật tại một ngưỡng

```python
threshold = 0.5
pred = (p_gbm >= threshold).astype(int)
print(classification_report(y_valid, pred, target_names=["stay", "churn"], digits=4))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (norm, fmt, title) in zip(axes, [(None, "d", "Số lượng"), ("true", ".1%", "Chuẩn hóa theo hàng = recall"),
                                         ("pred", ".1%", "Chuẩn hóa theo cột = precision")]):
    ConfusionMatrixDisplay.from_predictions(y_valid, pred, display_labels=["stay", "churn"], normalize=norm,
                                            values_format=fmt, cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"{title} @ {threshold}")
plt.tight_layout()
print("Ở ngưỡng 0.5 với prevalence ~16%: recall rất thấp -> ngưỡng mặc định không phù hợp")
```

### Đa lớp: confusion matrix và cách lấy trung bình

```python
from sklearn.datasets import load_wine
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Xw, yw = load_wine(return_X_y=True)
Xa, Xb, ya, yb = train_test_split(Xw, yw, test_size=0.4, stratify=yw, random_state=0)
yb_pred = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=1000)).fit(Xa, ya).predict(Xb)
print(confusion_matrix(yb, yb_pred))
print(multilabel_confusion_matrix(yb, yb_pred)[0])          # one-vs-rest 2x2 cho lớp 0
for avg in ["macro", "weighted", "micro"]:
    print(f"F1 {avg:8s} = {f1_score(yb, yb_pred, average=avg):.4f}")
```

| Kiểu trung bình | Cách tính | Khi nào dùng |
|---|---|---|
| **macro** | Trung bình đơn giản metric từng lớp | Mọi lớp quan trọng như nhau, kể cả lớp hiếm |
| **weighted** | Trung bình có trọng số theo support | Phản ánh phân phối dữ liệu; che giấu lớp hiếm kém |
| **micro** | Gộp TP/FP/FN toàn cục rồi tính | Đa nhãn; với đa lớp đơn nhãn, micro-F1 = accuracy |

**Đọc confusion matrix để phân tích lỗi:** hàng có nhiều giá trị ngoài đường chéo cho biết lớp đó khó nhận diện (recall thấp). Cột có nhiều giá trị ngoài đường chéo cho biết mô hình hay "đổ" về lớp đó (precision thấp). Cặp ô đối xứng lớn A↔B cho biết hai lớp nhầm lẫn lẫn nhau: cần feature phân biệt, hoặc gộp lớp nếu nghiệp vụ cho phép.

## 9.3. Metric không phụ thuộc ngưỡng

### ROC-AUC

Đường ROC vẽ TPR theo FPR khi quét mọi ngưỡng. **AUC bằng xác suất một mẫu positive ngẫu nhiên có điểm cao hơn một mẫu negative ngẫu nhiên.** Đây chính là thống kê Mann–Whitney U đã chuẩn hóa: $AUC = U/(n_1 n_0)$.

```python
from scipy.stats import mannwhitneyu

u = mannwhitneyu(p_gbm[y_valid.to_numpy() == 1], p_gbm[y_valid.to_numpy() == 0]).statistic
print("AUC =", round(roc_auc_score(y_valid, p_gbm), 6), "| U/(n1·n0) =",
      round(u / ((y_valid == 1).sum() * (y_valid == 0).sum()), 6))
```

### Precision–Recall curve và Average Precision

**[Docs]** `average_precision_score` tính $AP = \sum_n (R_n - R_{n-1})P_n$. Đây là tổng có trọng số của precision tại mỗi ngưỡng, **không nội suy tuyến tính** (nội suy hình thang trên đường PR cho kết quả lạc quan). **Baseline của AP bằng prevalence**, không phải 0.5.

```python
from sklearn.metrics import DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
for name, p in [("Logistic", p_logit), ("LightGBM", p_gbm)]:
    RocCurveDisplay.from_predictions(y_valid, p, name=name, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_valid, p, name=name, ax=axes[1])
    DetCurveDisplay.from_predictions(y_valid, p, name=name, ax=axes[2])
axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[1].axhline(y_valid.mean(), color="k", ls="--", lw=0.8, label="prevalence")
axes[2].set_title("DET: FNR vs FPR (thang probit)")
plt.tight_layout()
```

| Metric | Đo cái gì | Khi nào ưu tiên |
|---|---|---|
| ROC-AUC | Khả năng xếp hạng toàn cục | Lớp tương đối cân bằng; so sánh mô hình giữa các tập có prevalence khác nhau |
| **PR-AUC (AP)** | Xếp hạng, tập trung vào lớp positive | **Positive hiếm**, quan tâm vùng top |
| DET curve | FNR theo FPR trên thang probit | So sánh hệ thống phát hiện (sinh trắc học, gian lận) |
| **Log loss** | Chất lượng xác suất; phạt rất nặng dự đoán sai mà tự tin | Cần xác suất đúng; huấn luyện |
| **Brier** | MSE của xác suất ∈ [0, 1] | Dễ diễn giải hơn log loss |
| D² log loss / D² Brier | Tỷ lệ "giải thích" so với mô hình dự đoán tỷ lệ cơ sở | Giống R² cho phân loại (sklearn ≥ 1.5/1.7) |

**Phân rã Brier (Murphy, 1973):** Brier = **Reliability** (lỗi calibration, càng nhỏ càng tốt) − **Resolution** (khả năng phân biệt, càng lớn càng tốt) + **Uncertainty** (độ khó cố hữu của dữ liệu). **[Docs]** Chính vì vậy, Brier thấp hơn **chưa chắc** nghĩa là calibration tốt hơn. Có thể đó là mô hình phân biệt tốt hơn nhưng calibrate kém hơn.

```python
from sklearn.metrics import d2_brier_score, d2_log_loss_score


def brier_decomposition(y, p, n_bins: int = 10) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    bins = np.clip((pd.qcut(p, n_bins, labels=False, duplicates="drop")), 0, None)
    base = y.mean()
    rel = res = 0.0
    for b in np.unique(bins):
        m = bins == b
        w = m.mean()
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - base) ** 2
    return {"brier": brier_score_loss(y, p), "reliability": rel, "resolution": res,
            "uncertainty": base * (1 - base)}


for name, p in [("logit", p_logit), ("lightgbm", p_gbm)]:
    print(name, {k: round(v, 5) for k, v in brier_decomposition(y_valid, p).items()},
          "| D²(log loss)=", round(d2_log_loss_score(y_valid, np.c_[1 - p, p]), 4),
          "| D²(Brier)=", round(d2_brier_score(y_valid, p), 4))
```

### KS statistic và Gini (chuẩn ngành tín dụng)

```python
from scipy.stats import ks_2samp

ks = ks_2samp(p_gbm[y_valid == 1], p_gbm[y_valid == 0]).statistic
print(f"KS = {ks:.4f} | Gini = {2 * roc_auc_score(y_valid, p_gbm) - 1:.4f}")
```

## 9.4. Chọn ngưỡng quyết định

**Quan trọng:** chọn ngưỡng trên **validation hoặc OOF predictions**, rồi cố định và đánh giá **một lần** trên test.

### Quét ngưỡng với `metric_at_thresholds` (scikit-learn ≥ 1.9)

```python
from sklearn.metrics import metric_at_thresholds

oof = cross_val_predict(make_pipeline(build_preprocessor(scale=False), lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50, n_jobs=1, verbose=-1,
    random_state=0)),
    X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]

# metric_at_thresholds gọi metric tại MỌI giá trị điểm duy nhất (~20 000 ngưỡng -> rất chậm).
# Làm tròn điểm về 3 chữ số để chỉ còn ≤ 1 000 ngưỡng: đủ mịn cho quyết định kinh doanh.
oof_r = np.round(oof, 3)
f2_vals, thr = metric_at_thresholds(y_train, oof_r, fbeta_score, metric_params={"beta": 2})
mcc_vals, thr_mcc = metric_at_thresholds(y_train, oof_r, matthews_corrcoef)
print("Số ngưỡng được quét:", len(thr))
print(f"Ngưỡng tối đa F2  = {thr[np.argmax(f2_vals)]:.3f} (F2 = {f2_vals.max():.4f})")
print(f"Ngưỡng tối đa MCC = {thr_mcc[np.argmax(mcc_vals)]:.3f} (MCC = {mcc_vals.max():.4f})")
```

### Các chiến lược chọn ngưỡng

```python
prec, rec, thr_pr = precision_recall_curve(y_train, oof)        # len(thr_pr) = len(prec) - 1
f1_curve = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
fpr, tpr, thr_roc = roc_curve(y_train, oof)
ok = prec[:-1] >= 0.40                                           # ràng buộc: precision tối thiểu 40%


def expected_profit(y, p, t, value=500.0, retain=0.3, cost=50.0):
    """Ma trận lợi ích Chương 1.5: TP = v·r − c, FP = −c, FN = TN = 0 (nghìn VNĐ)."""
    act = p >= t
    y = np.asarray(y)
    return float(np.sum(act & (y == 1)) * (value * retain - cost) - np.sum(act & (y == 0)) * cost)


grid = np.linspace(0.02, 0.9, 89)
profits = np.array([expected_profit(y_train, oof, t) for t in grid])
strategies = {
    "mặc định 0.5": 0.5,
    "max F1": thr_pr[np.argmax(f1_curve)],
    "Youden J (ROC)": thr_roc[np.argmax(tpr - fpr)],
    "recall max | precision ≥ 0.40": thr_pr[ok][np.argmax(rec[:-1][ok])] if ok.any() else np.nan,
    "năng lực CSKH: top 10%": np.quantile(oof, 0.90),
    "max lợi nhuận kỳ vọng": grid[np.argmax(profits)],
    "lý thuyết c/(v·r) (cần calibrate)": 50 / (500 * 0.3),
}
rows = []
for name, t in strategies.items():
    pv = (p_gbm >= t).astype(int)
    rows.append({"strategy": name, "threshold": t, "precision": precision_score(y_valid, pv, zero_division=0),
                 "recall": recall_score(y_valid, pv), "f1": f1_score(y_valid, pv),
                 "pct_flagged": pv.mean(), "profit_valid": expected_profit(y_valid, p_gbm, t)})
print(pd.DataFrame(rows).set_index("strategy").round(3))
```

**Ngưỡng tối ưu theo chi phí (Elkan, 2001).** Với ma trận chi phí tổng quát và xác suất **đã calibrate**:

$$t^* = \frac{C_{FP} - C_{TN}}{(C_{FP} - C_{TN}) + (C_{FN} - C_{TP})}$$

Trường hợp churn ở Chương 1.5 cho $t^* = c/(v\,r) ≈ 0.333$.

### `TunedThresholdClassifierCV` và `FixedThresholdClassifier`

**[Docs]** `TunedThresholdClassifierCV` tìm ngưỡng tối ưu cho một `scoring` bằng **CV nội bộ** (mặc định 5-fold stratified). Lưu ý của tài liệu: **không bao giờ** dùng cùng dữ liệu để huấn luyện bộ phân loại và chọn ngưỡng. Muốn tối ưu lợi nhuận, dùng `make_scorer` với hàm lợi ích tự định nghĩa, như ví dụ chính thức *Post-hoc tuning the cut-off point of decision function* và *cost-sensitive learning*. `FixedThresholdClassifier` đóng gói một ngưỡng cố định vào mô hình để deploy nhất quán. Bọc `FrozenEstimator` nếu không muốn fit lại mô hình gốc.

```python
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import FixedThresholdClassifier, TunedThresholdClassifierCV


def profit_metric(y_true, y_pred, value=500.0, retain=0.3, cost=50.0):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sum((y_pred == 1) & (y_true == 1)) * (value * retain - cost)
                 - np.sum((y_pred == 1) & (y_true == 0)) * cost)


profit_scorer = make_scorer(profit_metric)
tuned = TunedThresholdClassifierCV(gbm, scoring=profit_scorer, thresholds=100,
                                   cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                   store_cv_results=True, random_state=0).fit(X_train, y_train)
print(f"Ngưỡng tối ưu lợi nhuận (CV nội bộ) = {tuned.best_threshold_:.3f}")
print(f"Lợi nhuận valid: mặc định 0.5 = {profit_metric(y_valid, gbm.predict(X_valid)):,.0f} | "
      f"tuned = {profit_metric(y_valid, tuned.predict(X_valid)):,.0f} (nghìn VNĐ)")

deployable = FixedThresholdClassifier(FrozenEstimator(gbm), threshold=float(tuned.best_threshold_),
                                      response_method="predict_proba").fit(X_train, y_train)
assert (deployable.predict(X_valid) == tuned.predict(X_valid)).mean() > 0.99
```

## 9.5. Metric xếp hạng kinh doanh: Lift, Gain, Precision@K

```python
lt = lift_table(y_valid, p_gbm, n_bins=10)
print(lt.round(3))


def precision_recall_at_k(y, p, k_frac: float) -> tuple[float, float]:
    y = np.asarray(y)
    top = np.argsort(-p)[: int(len(p) * k_frac)]
    return float(y[top].mean()), float(y[top].sum() / y.sum())


for k in [0.05, 0.10, 0.20]:
    pk, rk = precision_recall_at_k(y_valid, p_gbm, k)
    print(f"Top {k:.0%}: precision={pk:.3f} recall={rk:.3f} lift={pk / y_valid.mean():.2f}")
```

*Diễn giải cho stakeholder:* "Gọi 10% khách có điểm cao nhất sẽ tiếp cận được X% số khách sắp rời bỏ, hiệu quả gấp Y lần gọi ngẫu nhiên."

## 9.6. Calibration: xác suất có đáng tin không?

**[Docs]** Một bộ phân loại **calibrated tốt**: trong các mẫu có `predict_proba` ≈ 0.8, khoảng 80% thuộc lớp positive. Các dạng lệch điển hình theo tài liệu:

- **Logistic Regression** thường calibrate tốt khi đúng đặc tả và regularization phù hợp, nhờ *balance property* của canonical link.
- **Random Forest, bagging** có đường calibration hình chữ **S**: dự đoán bị kéo khỏi 0 và 1, vì phương sai của các cây thành phần khó tạo ra xác suất cực trị (Niculescu-Mizil & Caruana, 2005).
- **SVM / mô hình max-margin** cũng có dạng chữ S, do tập trung vào mẫu gần biên.
- **Naive Bayes** cho xác suất cực đoan do giả định độc lập.
- **Resampling / class weight** làm lệch toàn bộ xác suất (Chương 5.9).

### Các phương pháp calibration (scikit-learn)

| `method` | Mô hình calibrator | Khi nào dùng |
|---|---|---|
| `"sigmoid"` (Platt) | Logistic 1 chiều trên điểm số | Ít dữ liệu; lệch dạng chữ S |
| `"isotonic"` | Hàm bậc thang không giảm | Nhiều dữ liệu (tài liệu: isotonic tốt bằng hoặc hơn sigmoid khi đủ dữ liệu; dễ overfit khi ít); tạo **ties** nên có thể làm giảm nhẹ ROC-AUC |
| `"temperature"` (sklearn ≥ 1.8) | Chia logit cho T | **Đa lớp**, giữ nguyên argmax |

**[Docs]** Calibrator phải được fit trên dữ liệu **độc lập** với dữ liệu huấn luyện bộ phân loại. `CalibratedClassifierCV` lo việc này bằng CV:

- `ensemble=True` (mặc định khi không frozen): mỗi fold sinh một cặp (classifier, calibrator), dự đoán là trung bình. Thường **calibrate tốt hơn và chính xác hơn một chút**.
- `ensemble=False`: dùng `cross_val_predict` để fit một calibrator, một classifier trên toàn bộ dữ liệu. **Rẻ hơn** khi suy luận.
- Mô hình **đã huấn luyện**: bọc `FrozenEstimator(model)` và fit calibrator trên tập validation riêng. Cách này thay cho `cv="prefit"` (đã deprecate từ 1.6).

```python
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

# Mô hình bị lệch calibration có chủ đích: class_weight làm xác suất bị đẩy lên
skewed = make_pipeline(build_preprocessor(scale=False), lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50, class_weight="balanced",
    verbose=-1, random_state=0)).fit(X_train, y_train)

cal_sigmoid = CalibratedClassifierCV(FrozenEstimator(skewed), method="sigmoid").fit(X_valid, y_valid)
cal_isotonic = CalibratedClassifierCV(FrozenEstimator(skewed), method="isotonic").fit(X_valid, y_valid)

fig, ax = plt.subplots(figsize=(6.5, 6))
rows = []
for name, m in [("LightGBM chuẩn", gbm), ("class_weight=balanced", skewed),
                ("+ sigmoid", cal_sigmoid), ("+ isotonic", cal_isotonic)]:
    p = m.predict_proba(X_test)[:, 1]                       # đánh giá trên TEST, độc lập với calibrator
    CalibrationDisplay.from_predictions(y_test, p, n_bins=10, strategy="quantile", name=name, ax=ax)
    rows.append({"model": name, "mean_pred": p.mean(), "ECE": expected_calibration_error(y_test, p),
                 "brier": brier_score_loss(y_test, p), "log_loss": log_loss(y_test, p),
                 "roc_auc": roc_auc_score(y_test, p)})
ax.set_title("Reliability diagram (test)")
print("Tỷ lệ churn thật (test):", round(y_test.mean(), 4))
print(pd.DataFrame(rows).set_index("model").round(4))
```

> **Kết luận từ thí nghiệm:** `class_weight="balanced"` giữ nguyên khả năng xếp hạng (ROC-AUC gần như không đổi) nhưng làm xác suất trung bình lệch xa tỷ lệ thật, nên ECE, Brier và log loss tệ đi. Calibration sau huấn luyện khôi phục xác suất mà **không đổi thứ hạng**. Riêng isotonic có thể đổi nhẹ do tạo ties.

## 9.7. Metric hồi quy

**[Docs]** Bảng *strictly consistent scoring functions* của scikit-learn:

| Đại lượng cần dự đoán | Hàm đánh giá nhất quán | scikit-learn |
|---|---|---|
| Mean | Squared error (RMSE, R²) | `root_mean_squared_error`, `r2_score` |
| Mean (dữ liệu đếm) | Poisson deviance | `mean_poisson_deviance` |
| Mean (dương, lệch) | Gamma / Tweedie deviance | `mean_gamma_deviance`, `mean_tweedie_deviance` |
| Median | Absolute error | `mean_absolute_error` |
| Quantile α | Pinball loss | `mean_pinball_loss(alpha=α)` |

| Metric | Công thức | Đặc điểm |
|---|---|---|
| MAE | $\frac1n\sum\lvert y-\hat y\rvert$ | Cùng đơn vị; bền ngoại lai |
| RMSE | $\sqrt{\frac1n\sum(y-\hat y)^2}$ | Phạt nặng lỗi lớn |
| R² | $1 - SS_{res}/SS_{tot}$ | Có thể âm; phụ thuộc phương sai của tập đánh giá |
| MAPE | $\frac{100}{n}\sum\lvert(y-\hat y)/y\rvert$ | Vỡ khi y ≈ 0; bất đối xứng (phạt over-forecast nhẹ hơn) |
| sMAPE | $\frac{100}{n}\sum\frac{2\lvert y-\hat y\rvert}{\lvert y\rvert+\lvert\hat y\rvert}$ | Đối xứng hơn |
| WAPE | $\sum\lvert y-\hat y\rvert/\sum\lvert y\rvert$ | Chuẩn ngành bán lẻ |
| MASE | MAE / MAE(naive) | Chuỗi thời gian; < 1 là tốt hơn naive (Hyndman & Koehler, 2006) |
| Pinball | $\max(\alpha(y-\hat y), (\alpha-1)(y-\hat y))$ | Dự báo phân vị |

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, mean_pinball_loss,
                             mean_poisson_deviance, r2_score, root_mean_squared_error)

rng = np.random.default_rng(0)
n = 6_000
Xr = pd.DataFrame({"tenure": rng.integers(1, 72, n), "plan": rng.integers(0, 3, n), "calls": rng.poisson(1.5, n)})
y_rev = rng.gamma(shape=2.0, scale=np.exp(3 + 0.02 * Xr["tenure"] + 0.3 * Xr["plan"]) / 2.0)
Xa, Xb, ya, yb = train_test_split(Xr, y_rev, test_size=0.3, random_state=0)

models = {
    "squared_error (mean)": HistGradientBoostingRegressor(loss="squared_error", random_state=0),
    "absolute_error (median)": HistGradientBoostingRegressor(loss="absolute_error", random_state=0),
    "gamma (mean, dương)": HistGradientBoostingRegressor(loss="gamma", random_state=0),
}
rows = []
for name, m in models.items():
    pred = m.fit(Xa, ya).predict(Xb)
    rows.append({"loss": name, "RMSE": root_mean_squared_error(yb, pred), "MAE": mean_absolute_error(yb, pred),
                 "R2": r2_score(yb, pred), "MAPE_%": 100 * mean_absolute_percentage_error(yb, pred),
                 "PoissonDev": mean_poisson_deviance(yb, np.clip(pred, 1e-6, None)),
                 "sum_pred/sum_true": pred.sum() / yb.sum()})
print(pd.DataFrame(rows).set_index("loss").round(3))   # mỗi loss thắng ở metric nhất quán với nó

# Dự báo khoảng 80% bằng hai mô hình phân vị + kiểm tra coverage
lo = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, random_state=0).fit(Xa, ya).predict(Xb)
hi = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, random_state=0).fit(Xa, ya).predict(Xb)
print(f"Coverage khoảng [P10, P90] = {np.mean((yb >= lo) & (yb <= hi)):.3f} (mục tiêu 0.80) | "
      f"pinball@0.9 = {mean_pinball_loss(yb, hi, alpha=0.9):.3f}")
# Hồi quy phân vị KHÔNG bảo đảm coverage trên dữ liệu mới (thường thiếu hụt) -> cần kiểm tra, hoặc dùng conformal
```

### Conformal prediction: khoảng dự đoán có bảo đảm coverage

**Split conformal** cho bảo đảm $P(y \in \hat C(x)) \ge 1-\alpha$ dưới giả định **exchangeability**, không cần giả định phân phối. Thư viện chuyên dụng: **MAPIE**.

```python
Xfit, Xcal, yfit, ycal = train_test_split(Xa, ya, test_size=0.3, random_state=1)
point = HistGradientBoostingRegressor(random_state=0).fit(Xfit, yfit)
scores = np.abs(ycal - point.predict(Xcal))                       # nonconformity score
alpha = 0.2
q = np.quantile(scores, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores), method="higher")
pred_b = point.predict(Xb)
print(f"Split conformal 80%: coverage={np.mean(np.abs(yb - pred_b) <= q):.3f} | độ rộng khoảng={2 * q:.1f}")
```

## 9.8. Metric phân cụm

```python
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)

Xc = StandardScaler().fit_transform(X_train[["tenure_months", "monthly_charges", "support_calls"]])
rows = []
for k in range(2, 8):
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xc)
    rows.append({"k": k, "silhouette↑": silhouette_score(Xc, labels, sample_size=4000, random_state=0),
                 "davies_bouldin↓": davies_bouldin_score(Xc, labels),
                 "calinski_harabasz↑": calinski_harabasz_score(Xc, labels)})
print(pd.DataFrame(rows).set_index("k").round(3))
true_groups = X_train["contract"].astype("category").cat.codes
km3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Xc)
print("ARI so với contract:", round(adjusted_rand_score(true_groups, km3), 4),
      "| NMI:", round(normalized_mutual_info_score(true_groups, km3), 4))
# ARI ≈ 0: các cụm (dựa trên tenure/cước/cuộc gọi) không trùng với loại hợp đồng -> phân cụm tìm cấu trúc khác
```

Metric nội tại chỉ là gợi ý. Phân cụm **có giá trị khi các cụm khác biệt về hành vi và hành động được**. Hãy đánh giá bằng profile cụm và sự đồng thuận của nghiệp vụ.

## 9.9. Metric truy hồi & xếp hạng (search, RAG, recommender)

```python
from sklearn.metrics import ndcg_score


def recall_at_k(relevant: set, ranked: list, k: int) -> float:
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def average_precision_at_k(relevant: set, ranked: list, k: int) -> float:
    hits, score = 0, 0.0
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0


def reciprocal_rank(relevant: set, ranked: list) -> float:
    return next((1.0 / i for i, d in enumerate(ranked, 1) if d in relevant), 0.0)


def f_beta_sets(relevant: set, retrieved: set, beta: float = 2.0) -> float:
    """F2 ưu tiên recall (trọng số β² = 4): phù hợp truy hồi văn bản pháp luật, nơi bỏ sót đắt hơn thừa."""
    tp = len(relevant & retrieved)
    if tp == 0:
        return 0.0
    p, r = tp / len(retrieved), tp / len(relevant)
    return (1 + beta**2) * p * r / (beta**2 * p + r)


queries = [({"d1", "d4"}, ["d4", "d2", "d1", "d7"]), ({"d3"}, ["d5", "d3", "d9", "d1"])]
print("Recall@3 :", np.mean([recall_at_k(r, ranked, 3) for r, ranked in queries]))
print("MAP@3    :", np.mean([average_precision_at_k(r, ranked, 3) for r, ranked in queries]).round(4))
print("MRR      :", np.mean([reciprocal_rank(r, ranked) for r, ranked in queries]))
print("F2 (top3):", np.mean([f_beta_sets(r, set(ranked[:3])) for r, ranked in queries]).round(4))
print("NDCG@3   :", round(ndcg_score([[3, 2, 0, 0, 1]], [[0.9, 0.8, 0.7, 0.1, 0.6]], k=3), 4))
```

## 9.10. So sánh mô hình có ý nghĩa thống kê

Trên validation, Logistic có PR-AUC ≈ 0.375 và LightGBM ≈ 0.369. Chênh lệch nhỏ như vậy có thật không, hay chỉ do nhiễu?

```python
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from statsmodels.stats.contingency_tables import mcnemar


def paired_bootstrap(y, p_a, p_b, metric=average_precision_score, n_boot=2000, seed=0) -> dict:
    """CI cho metric(B) − metric(A) trên CÙNG tập test (ghép cặp)."""
    r = np.random.default_rng(seed)
    y = np.asarray(y)
    diffs = []
    for _ in range(n_boot):
        i = r.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            diffs.append(metric(y[i], p_b[i]) - metric(y[i], p_a[i]))
    diffs = np.asarray(diffs)
    return {"diff": float(metric(y, p_b) - metric(y, p_a)), "ci95": np.round(np.quantile(diffs, [0.025, 0.975]), 4),
            "P(B>A)": float((diffs > 0).mean())}


def corrected_resampled_ttest(scores_a, scores_b, n_train: int, n_test: int, k: int) -> tuple[float, float]:
    """Nadeau & Bengio (2003): hiệu chỉnh phương sai vì các fold CV không độc lập."""
    d = np.asarray(scores_b) - np.asarray(scores_a)
    var = d.var(ddof=1) * (1 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


p_logit_te, p_gbm_te = logit.predict_proba(X_test)[:, 1], gbm.predict_proba(X_test)[:, 1]
print("Paired bootstrap PR-AUC (LightGBM − Logistic):", paired_bootstrap(y_test, p_logit_te, p_gbm_te))

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
s_logit = cross_val_score(logit, X_train, y_train, cv=rcv, scoring="average_precision", n_jobs=-1)
s_gbm = cross_val_score(gbm, X_train, y_train, cv=rcv, scoring="average_precision", n_jobs=-1)
naive_p = stats.ttest_rel(s_gbm, s_logit).pvalue
t, p_corr = corrected_resampled_ttest(s_logit, s_gbm, n_train=int(len(X_train) * 0.8),
                                      n_test=int(len(X_train) * 0.2), k=len(s_gbm))
print(f"t-test ghép cặp NGÂY THƠ p={naive_p:.4f} | corrected resampled t-test p={p_corr:.4f}")

a_ok = (p_logit_te >= 0.3) == y_test.to_numpy()
b_ok = (p_gbm_te >= 0.3) == y_test.to_numpy()
table = [[np.sum(a_ok & b_ok), np.sum(a_ok & ~b_ok)], [np.sum(~a_ok & b_ok), np.sum(~a_ok & ~b_ok)]]
print("McNemar (nhãn cứng @0.3) p =", round(mcnemar(table, exact=False, correction=True).pvalue, 4))
```

**Đọc kết quả:** bootstrap ghép cặp trên test cho CI của chênh lệch PR-AUC (LightGBM − Logistic) **nằm hoàn toàn dưới 0**, nên trên dữ liệu gần tuyến tính này Logistic tốt hơn thật. t-test ngây thơ trên các fold CV cho p ≈ 0. Corrected t-test vẫn có ý nghĩa nhưng p lớn hơn hàng trăm lần, vì t-test ngây thơ đánh giá thấp phương sai. McNemar trên nhãn cứng tại ngưỡng 0.3 **không** có ý nghĩa: hai mô hình đưa ra quyết định gần như giống nhau ở ngưỡng đó. Mỗi kiểm định trả lời một câu hỏi khác nhau.

**[Docs]** Ví dụ chính thức *Statistical comparison of models using grid search* của scikit-learn trình bày cả cách tiếp cận tần suất (corrected t-test) và Bayes (xác suất một mô hình tốt hơn, *ROPE*: region of practical equivalence). Nguyên tắc: nếu CI của chênh lệch chứa 0 hoặc nằm trong vùng tương đương thực tế, **giữ mô hình đơn giản hơn**.

## 9.11. Phân tích theo phân khúc (slice analysis) và phân tích lỗi

```python
def slice_report(X: pd.DataFrame, y, p, col: str, threshold: float) -> pd.DataFrame:
    d = X[[col]].copy()
    d["y"], d["p"] = np.asarray(y), p
    d["pred"] = (d["p"] >= threshold).astype(int)
    rows = []
    for key, g in d.groupby(col, observed=True, dropna=False):
        if g["y"].nunique() < 2 or len(g) < 50:
            continue
        rows.append({col: key, "n": len(g), "prevalence": g["y"].mean(), "roc_auc": roc_auc_score(g["y"], g["p"]),
                     "pr_auc": average_precision_score(g["y"], g["p"]), "recall": recall_score(g["y"], g["pred"]),
                     "precision": precision_score(g["y"], g["pred"], zero_division=0),
                     "calib_gap": g["p"].mean() - g["y"].mean()})
    return pd.DataFrame(rows).sort_values("pr_auc")


t_star = float(tuned.best_threshold_)
print(slice_report(X_valid, y_valid, p_gbm, "contract", t_star).round(3).to_string(index=False))
X_slices = X_valid.assign(tenure_band=pd.cut(X_valid["tenure_months"], [0, 6, 24, 48, 72]))
print(slice_report(X_slices, y_valid, p_gbm, "tenure_band", t_star).round(3).to_string(index=False))

errors = X_valid.assign(y=y_valid.to_numpy(), p=p_gbm)
print("FN tự tin nhất (churn thật, điểm thấp):")
print(errors[errors.y == 1].nsmallest(5, "p")[["contract", "tenure_months", "support_calls", "monthly_charges", "p"]])
```

## 9.12. Báo cáo đánh giá cuối cùng trên tập test

```python
from churn.models.evaluate import bootstrap_metric_ci

final_p = gbm.predict_proba(X_test)[:, 1]
final_pred = (final_p >= t_star).astype(int)
metrics = {"ROC-AUC": roc_auc_score, "PR-AUC": average_precision_score, "Brier": brier_score_loss}
rows = []
for name, fn in metrics.items():
    point, lo, hi = bootstrap_metric_ci(y_test, final_p, fn, n_boot=500)
    rows.append({"metric": name, "value": point, "ci95_low": lo, "ci95_high": hi})
for name, fn in {"Recall": recall_score, "Precision": precision_score}.items():
    point, lo, hi = bootstrap_metric_ci(y_test, final_pred, fn, n_boot=500)
    rows.append({"metric": f"{name} @ {t_star:.2f}", "value": point, "ci95_low": lo, "ci95_high": hi})
print(pd.DataFrame(rows).set_index("metric").round(4))
print("Lợi nhuận kỳ vọng trên test (nghìn VNĐ):", f"{expected_profit(y_test, final_p, t_star):,.0f}")
```

> **Checklist Chương 9**
> - [ ] Tách đánh giá **xác suất** (log loss, Brier, calibration) và **quyết định** (ngưỡng, confusion matrix, lợi nhuận).
> - [ ] Không dùng Accuracy làm metric chính khi mất cân bằng; nhớ precision phụ thuộc prevalence.
> - [ ] Confusion matrix phân tích tại **ngưỡng thực sự triển khai**, theo cả hàng (recall) và cột (precision).
> - [ ] Ngưỡng chọn trên validation/OOF theo chi phí hoặc năng lực vận hành; đóng gói bằng `FixedThresholdClassifier`.
> - [ ] Kiểm tra calibration (reliability diagram, ECE); calibrate bằng `CalibratedClassifierCV` (+ `FrozenEstimator`) trên dữ liệu độc lập.
> - [ ] Hồi quy: metric nhất quán với đại lượng dự đoán; dự báo khoảng có kiểm tra coverage.
> - [ ] So sánh mô hình bằng kiểm định ghép cặp / corrected t-test; báo cáo CI.
> - [ ] Slice analysis cho các phân khúc quan trọng; xem các lỗi tự tin nhất.

### Tài liệu tham khảo Chương 9

- scikit-learn User Guide: *Metrics and scoring: quantifying the quality of predictions* (Which scoring function should I use?; classification, regression, ranking metrics); *Tuning the decision threshold for class prediction*; *Probability calibration*.
- scikit-learn Examples: *Post-tuning the decision threshold for cost-sensitive learning*; *Probability Calibration curves*; *Statistical comparison of models using grid search*; *Prediction Intervals for Gradient Boosting Regression*.
- Gneiting, T. & Raftery, A. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA 102. · Gneiting, T. (2011). *Making and Evaluating Point Forecasts.* JASA 106.
- Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning.* IJCAI.
- Saito, T. & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLOS ONE.
- Chicco, D. & Jurman, G. (2020). *The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy.* BMC Genomics.
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning.* ICML.
- Murphy, A. H. (1973). *A New Vector Partition of the Probability Score.* J. Applied Meteorology.
- Nadeau, C. & Bengio, Y. (2003). *Inference for the Generalization Error.* Machine Learning 52. · Dietterich, T. (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.* Neural Computation.
- Angelopoulos, A. & Bates, S. (2023). *Conformal Prediction: A Gentle Introduction.* Foundations and Trends in ML. · MAPIE documentation.
- Hyndman, R. & Koehler, A. (2006). *Another look at measures of forecast accuracy.* IJF 22(4).
- Google ML Crash Course: *Classification: Accuracy, precision, recall; ROC and AUC; Prediction bias*.
