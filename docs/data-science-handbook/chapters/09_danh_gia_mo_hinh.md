# CHƯƠNG 9. ĐÁNH GIÁ MÔ HÌNH: METRICS, CONFUSION MATRIX, THRESHOLD & CALIBRATION

## 9.1. Nguyên tắc chọn metric

1. **Metric phải phản ánh chi phí kinh doanh** (Chương 1.3), không phải metric "phổ biến".
2. Tách biệt: **metric tối ưu (optimizing)** — một con số để chọn mô hình; và **metric ràng buộc (satisficing)** — phải đạt ngưỡng (ví dụ: độ trễ < 50ms, recall ≥ 0.7, chênh lệch fairness < 5%).
3. Báo cáo **khoảng tin cậy**, không chỉ ước lượng điểm.
4. Đánh giá **theo phân khúc (slice)**: mô hình tốt trung bình có thể rất tệ với một nhóm khách hàng.

### Bảng tra nhanh

| Bài toán | Metric chính | Metric bổ sung |
|---|---|---|
| Phân loại cân bằng | Accuracy, ROC-AUC, F1 | Log loss |
| Phân loại mất cân bằng | **PR-AUC**, Recall@Precision, F-β | MCC, Balanced accuracy |
| Cần xác suất chính xác (định giá rủi ro) | **Log loss, Brier score** | ECE (calibration) |
| Xếp hạng top-K (marketing, CSKH) | Precision@K, Recall@K, Lift@K | Gain chart |
| Tín dụng | Gini (= 2·AUC − 1), KS | PSI (ổn định) |
| Hồi quy | MAE, RMSE | R², MAPE/sMAPE, pinball loss |
| Truy hồi thông tin / gợi ý | NDCG@K, MAP@K, MRR, Recall@K | F2 (ưu tiên recall) |
| Phân cụm | Silhouette, Davies–Bouldin | ARI/NMI (khi có nhãn) |

## 9.2. Confusion Matrix — nền tảng của mọi metric phân loại

```text
                         DỰ ĐOÁN
                    Positive (1)        Negative (0)
            ┌────────────────────┬────────────────────┐
 THỰC  P(1) │  TP (True Positive)│ FN (False Negative)│  ← Sai lầm loại II (bỏ sót)
 TẾ         │  Đúng: churn       │ Bỏ sót khách churn │
            ├────────────────────┼────────────────────┤
       N(0) │ FP (False Positive)│  TN (True Negative)│
            │ Báo nhầm (tốn tiền)│  Đúng: không churn │  ← FP = Sai lầm loại I
            └────────────────────┴────────────────────┘
```

> Lưu ý quy ước: `sklearn.metrics.confusion_matrix` trả về ma trận với **hàng = thực tế, cột = dự đoán**, theo thứ tự nhãn tăng dần `[0, 1]`, tức là `[[TN, FP], [FN, TP]]` — ngược thứ tự so với hình trên.

### Các chỉ số dẫn xuất

| Chỉ số | Công thức | Ý nghĩa | Tên khác |
|---|---|---|---|
| **Accuracy** | $\dfrac{TP + TN}{TP + TN + FP + FN}$ | Tỷ lệ dự đoán đúng | — vô dụng khi mất cân bằng |
| **Precision** | $\dfrac{TP}{TP + FP}$ | Trong số dự đoán positive, bao nhiêu đúng? | PPV |
| **Recall** | $\dfrac{TP}{TP + FN}$ | Trong số positive thật, bắt được bao nhiêu? | Sensitivity, TPR, Hit rate |
| **Specificity** | $\dfrac{TN}{TN + FP}$ | Trong số negative thật, nhận đúng bao nhiêu? | TNR, Selectivity |
| FPR | $\dfrac{FP}{FP + TN} = 1 - \text{Specificity}$ | Tỷ lệ báo động nhầm | Fall-out |
| FNR | $\dfrac{FN}{FN + TP} = 1 - \text{Recall}$ | Tỷ lệ bỏ sót | Miss rate |
| NPV | $\dfrac{TN}{TN + FN}$ | Dự đoán negative đáng tin đến đâu? | |
| **F1** | $2 \cdot \dfrac{P \cdot R}{P + R}$ | Trung bình điều hòa P và R | |
| **F-β** | $(1 + \beta^2) \dfrac{P \cdot R}{\beta^2 P + R}$ | β > 1 ưu tiên Recall (F2), β < 1 ưu tiên Precision (F0.5) | |
| **Balanced Accuracy** | $\dfrac{TPR + TNR}{2}$ | Accuracy công bằng giữa các lớp | |
| **MCC** | $\dfrac{TP\cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | Tương quan dự đoán–thực tế, ∈ [−1, 1]; **bền nhất khi mất cân bằng** | Phi coefficient |
| **Cohen's κ** | $\dfrac{p_o - p_e}{1 - p_e}$ | Đồng thuận vượt mức ngẫu nhiên | |
| Youden's J | $TPR - FPR$ | Dùng chọn ngưỡng trên ROC | Informedness |

### Ví dụ tính tay

Tập kiểm tra 1000 khách, 200 churn. Mô hình dự đoán 250 churn, trong đó đúng 150.

| | Dự đoán 1 | Dự đoán 0 | Tổng |
|---|---|---|---|
| Thực tế 1 | TP = 150 | FN = 50 | 200 |
| Thực tế 0 | FP = 100 | TN = 700 | 800 |

- Accuracy = (150 + 700)/1000 = **0.85**
- Precision = 150/250 = **0.60**; Recall = 150/200 = **0.75**; Specificity = 700/800 = **0.875**
- F1 = 2·0.6·0.75/(1.35) = **0.667**; F2 = 5·0.6·0.75/(4·0.6 + 0.75) = **0.714**
- MCC = (150·700 − 100·50)/√(250·200·800·750) = 100000/173205 = **0.577**
- Mô hình "luôn dự đoán 0" có Accuracy = **0.80** nhưng Recall = 0, MCC = 0 → minh chứng Accuracy gây hiểu lầm.

### Code: tính và trực quan hóa confusion matrix

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score, fbeta_score,
                             matthews_corrcoef, precision_score, recall_score)

# y_valid, proba_valid: nhãn thật và xác suất dự đoán của mô hình (chương 8)
proba_valid = pipe.predict_proba(X_valid)[:, 1]
threshold = 0.5
y_pred = (proba_valid >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_valid, y_pred, labels=[0, 1]).ravel()


def binary_report(y_true, y_pred) -> pd.Series:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return pd.Series({
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "npv": tn / (tn + fn) if (tn + fn) else 0.0,
        "f1": f1_score(y_true, y_pred),
        "f2": fbeta_score(y_true, y_pred, beta=2),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    })


print(binary_report(y_valid, y_pred).round(4))
print(classification_report(y_valid, y_pred, target_names=["stay", "churn"], digits=4))

# Vẽ 2 phiên bản: số tuyệt đối & chuẩn hóa theo hàng (= recall từng lớp)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay.from_predictions(y_valid, y_pred, display_labels=["stay", "churn"],
                                        cmap="Blues", values_format="d", ax=axes[0])
axes[0].set_title("Confusion matrix (counts)")
ConfusionMatrixDisplay.from_predictions(y_valid, y_pred, display_labels=["stay", "churn"],
                                        normalize="true", cmap="Blues", values_format=".2%",
                                        ax=axes[1])
axes[1].set_title("Normalized by true class (recall)")
plt.tight_layout()
```

`normalize` có 3 chế độ: `"true"` (chia theo hàng → recall từng lớp), `"pred"` (chia theo cột → precision từng lớp), `"all"` (chia tổng).

### Confusion matrix đa lớp & cách lấy trung bình

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

Xi, yi = load_iris(return_X_y=True)
Xa, Xb, ya, yb = train_test_split(Xi, yi, test_size=0.4, stratify=yi, random_state=0)
yb_pred = LogisticRegression(max_iter=1000).fit(Xa, ya).predict(Xb)

print(confusion_matrix(yb, yb_pred))
print(multilabel_confusion_matrix(yb, yb_pred))   # 1 ma trận 2x2 (one-vs-rest) cho mỗi lớp

for avg in ["macro", "weighted", "micro"]:
    print(avg, round(f1_score(yb, yb_pred, average=avg), 4))
```

| Kiểu trung bình | Cách tính | Khi nào dùng |
|---|---|---|
| **macro** | Trung bình đơn giản metric từng lớp | Mọi lớp quan trọng như nhau (kể cả lớp hiếm) |
| **weighted** | Trung bình có trọng số theo số mẫu (support) | Phản ánh phân phối dữ liệu |
| **micro** | Gộp TP/FP/FN toàn cục rồi tính | Đa nhãn; với đa lớp đơn nhãn, micro-F1 = accuracy |

### Đọc confusion matrix để phân tích lỗi

- **Hàng có nhiều giá trị ngoài đường chéo** → lớp đó khó nhận diện (recall thấp).
- **Cột có nhiều giá trị ngoài đường chéo** → mô hình hay "đổ" về lớp đó (precision thấp).
- **Cặp ô đối xứng lớn** (A↔B) → hai lớp bị nhầm lẫn lẫn nhau → cần feature phân biệt, hoặc gộp lớp nếu nghiệp vụ cho phép.

## 9.3. Metric không phụ thuộc ngưỡng

### ROC curve & ROC-AUC

ROC vẽ TPR theo FPR khi quét mọi ngưỡng. **AUC = xác suất mô hình xếp một mẫu positive ngẫu nhiên cao hơn một mẫu negative ngẫu nhiên.**

### Precision-Recall curve & PR-AUC (Average Precision)

Khi positive hiếm (< 10%), ROC-AUC có thể cao "giả" vì FPR bị pha loãng bởi số lượng TN khổng lồ. **PR-AUC tập trung vào lớp positive** — baseline của PR-AUC là tỷ lệ positive (không phải 0.5).

```python
from sklearn.metrics import (PrecisionRecallDisplay, RocCurveDisplay, average_precision_score,
                             brier_score_loss, log_loss, roc_auc_score)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
RocCurveDisplay.from_predictions(y_valid, proba_valid, name="LightGBM", ax=axes[0],
                                 plot_chance_level=True)
PrecisionRecallDisplay.from_predictions(y_valid, proba_valid, name="LightGBM", ax=axes[1],
                                        plot_chance_level=True)
plt.tight_layout()

auc = roc_auc_score(y_valid, proba_valid)
print({
    "roc_auc": auc,
    "gini": 2 * auc - 1,
    "pr_auc": average_precision_score(y_valid, proba_valid),
    "log_loss": log_loss(y_valid, proba_valid),
    "brier": brier_score_loss(y_valid, proba_valid),
})
```

| Metric | Đo cái gì | Ghi chú |
|---|---|---|
| ROC-AUC | Khả năng xếp hạng | Không nhạy với mất cân bằng — vừa là ưu vừa là nhược |
| PR-AUC | Xếp hạng tập trung vào positive | Ưu tiên khi lớp hiếm |
| Log loss | Chất lượng xác suất | Phạt rất nặng dự đoán sai mà tự tin |
| Brier score | MSE của xác suất | Dễ diễn giải hơn log loss, ∈ [0, 1] |

### KS statistic (tín dụng)

```python
from scipy.stats import ks_2samp

ks = ks_2samp(proba_valid[y_valid == 1], proba_valid[y_valid == 0]).statistic
print("KS =", round(ks, 4))   # Khoảng cách lớn nhất giữa 2 CDF; > 0.4 thường được coi là tốt
```

## 9.4. Chọn ngưỡng quyết định (Threshold Tuning)

Ngưỡng 0.5 chỉ tối ưu khi chi phí FP = FN **và** xác suất được calibrate tốt. Trong thực tế gần như không bao giờ như vậy.

**Quan trọng:** chọn ngưỡng trên **validation hoặc OOF predictions**, rồi cố định và đánh giá trên test.

```python
from sklearn.metrics import precision_recall_curve, roc_curve


def threshold_table(y_true, proba, thresholds=np.linspace(0.05, 0.95, 19)) -> pd.DataFrame:
    rows = [binary_report(y_true, (proba >= t).astype(int)).rename(round(t, 2)) for t in thresholds]
    return pd.DataFrame(rows)[["precision", "recall", "f1", "f2", "mcc", "TP", "FP", "FN"]]


print(threshold_table(y_valid, proba_valid).round(3))

# (1) Ngưỡng tối đa F1 / F-beta
prec, rec, thr = precision_recall_curve(y_valid, proba_valid)
f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)   # len(thr) = len(prec)-1
t_f1 = thr[np.argmax(f1)]

# (2) Youden's J trên ROC
fpr, tpr, thr_roc = roc_curve(y_valid, proba_valid)
t_youden = thr_roc[np.argmax(tpr - fpr)]

# (3) Ràng buộc nghiệp vụ: Precision tối thiểu 0.5, tối đa hóa Recall
ok = prec[:-1] >= 0.5
t_prec = thr[ok][np.argmax(rec[:-1][ok])] if ok.any() else None

# (4) Năng lực vận hành: CSKH chỉ gọi được top 10% khách
t_topk = np.quantile(proba_valid, 0.90)


# (5) Tối đa hóa lợi nhuận kỳ vọng (ma trận chi phí chương 1)
def expected_profit(y_true, proba, t, gain_tp=450_000, cost_fp=-50_000, cost_fn=-500_000):
    pred = proba >= t
    tp = np.sum(pred & (y_true == 1)); fp = np.sum(pred & (y_true == 0))
    fn = np.sum(~pred & (y_true == 1))
    return tp * gain_tp + fp * cost_fp + fn * cost_fn


grid = np.linspace(0.01, 0.99, 99)
profits = [expected_profit(y_valid.to_numpy(), proba_valid, t) for t in grid]
t_profit = grid[int(np.argmax(profits))]
print(dict(f1=t_f1, youden=t_youden, precision_constraint=t_prec, top10=t_topk, profit=t_profit))
```

Khi xác suất đã được calibrate, ngưỡng tối ưu lý thuyết theo chi phí là:

$$t^* = \frac{C_{FP}}{C_{FP} + C_{FN}} \quad(\text{chi phí tính theo giá trị dương, với } C_{TP} = C_{TN} = 0)$$

### sklearn ≥ 1.5: `TunedThresholdClassifierCV`

```python
from sklearn.metrics import make_scorer
from sklearn.model_selection import FixedThresholdClassifier, TunedThresholdClassifierCV

tuned = TunedThresholdClassifierCV(
    pipe, scoring=make_scorer(fbeta_score, beta=2), cv=5, thresholds=100,
    store_cv_results=True, random_state=0,
).fit(X_train, y_train)
print("Best threshold:", tuned.best_threshold_, "F2:", tuned.best_score_)

# Đóng gói ngưỡng cố định vào mô hình để deploy nhất quán
deployable = FixedThresholdClassifier(pipe, threshold=float(t_profit), response_method="predict_proba")
```

## 9.5. Metric xếp hạng / kinh doanh: Lift & Gain

```python
def lift_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y_true), "p": proba})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), n_bins,
                          labels=range(1, n_bins + 1))
    t = d.groupby("decile", observed=True).agg(n=("y", "size"), positives=("y", "sum"),
                                               min_score=("p", "min"))
    t["rate"] = t["positives"] / t["n"]
    t["lift"] = t["rate"] / d["y"].mean()
    t["cum_capture"] = t["positives"].cumsum() / d["y"].sum()     # Gain
    return t.round(4)


print(lift_table(y_valid, proba_valid))


def precision_at_k(y_true, proba, k: float = 0.1) -> float:
    n = int(len(proba) * k)
    idx = np.argsort(-proba)[:n]
    return float(np.asarray(y_true)[idx].mean())
```

*Diễn giải cho stakeholder:* "Nếu gọi 10% khách có điểm cao nhất, ta tiếp cận được 35% số khách sẽ rời bỏ, hiệu quả gấp 3.5 lần gọi ngẫu nhiên."

## 9.6. Calibration — xác suất có đáng tin không?

Mô hình **calibrated**: trong các khách được dự đoán 30% churn, thực tế ~30% churn. Quan trọng khi xác suất được dùng trực tiếp (tính lợi nhuận kỳ vọng, định giá, kết hợp nhiều mô hình).

Nguyên nhân mất calibration: class weight/SMOTE, mô hình boosting/SVM/Naive Bayes, overfitting.

```python
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay, calibration_curve


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(np.asarray(y_true)[m].mean() - proba[m].mean())
    return ece


# Calibrate: method="sigmoid" (Platt — ít dữ liệu) hoặc "isotonic" (nhiều dữ liệu, > ~1000 positive)
calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv=5).fit(X_train, y_train)
proba_cal = calibrated.predict_proba(X_valid)[:, 1]

fig, ax = plt.subplots(figsize=(6, 6))
CalibrationDisplay.from_predictions(y_valid, proba_valid, n_bins=10, strategy="quantile",
                                    name="Uncalibrated", ax=ax)
CalibrationDisplay.from_predictions(y_valid, proba_cal, n_bins=10, strategy="quantile",
                                    name="Isotonic", ax=ax)
print("ECE before:", round(expected_calibration_error(y_valid, proba_valid), 4),
      "| after:", round(expected_calibration_error(y_valid, proba_cal), 4))
print("Brier before/after:", brier_score_loss(y_valid, proba_valid), brier_score_loss(y_valid, proba_cal))
```

> Calibration không thay đổi thứ hạng nhiều (AUC gần như giữ nguyên) nhưng cải thiện log loss/Brier và làm ngưỡng có ý nghĩa.

## 9.7. Metric hồi quy

| Metric | Công thức | Đặc điểm |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum\lvert y - \hat y\rvert$ | Cùng đơn vị với y; bền với ngoại lai; tối ưu bởi median |
| **MSE / RMSE** | $\sqrt{\frac{1}{n}\sum(y - \hat y)^2}$ | Phạt nặng sai số lớn; tối ưu bởi mean |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Tỷ lệ phương sai được giải thích; có thể âm |
| Adjusted R² | $1 - (1 - R^2)\frac{n - 1}{n - p - 1}$ | Phạt số lượng feature |
| **MAPE** | $\frac{100}{n}\sum\left\lvert\frac{y - \hat y}{y}\right\rvert$ | Dễ hiểu (%); **vỡ khi y ≈ 0**, bất đối xứng |
| sMAPE | $\frac{100}{n}\sum\frac{2\lvert y - \hat y\rvert}{\lvert y\rvert + \lvert\hat y\rvert}$ | Đối xứng hơn MAPE |
| RMSLE | RMSE trên $\log(1+y)$ | Sai số tương đối; target lệch phải |
| MASE | MAE / MAE của naive forecast | Chuỗi thời gian; < 1 là tốt hơn naive |
| Pinball (quantile) loss | $\max(q(y-\hat y), (q-1)(y-\hat y))$ | Đánh giá dự báo phân vị / khoảng |
| WAPE | $\sum\lvert y - \hat y\rvert / \sum\lvert y\rvert$ | Dự báo nhu cầu bán lẻ |

```python
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_pinball_loss, mean_squared_log_error, r2_score,
                             root_mean_squared_error)


def regression_report(y_true, y_pred, n_features: int | None = None) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    r2 = r2_score(y_true, y_pred)
    out = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),      # sklearn >= 1.4
        "R2": r2,
        "MAPE_%": 100 * mean_absolute_percentage_error(y_true, y_pred),
        "sMAPE_%": 100 * np.mean(2 * np.abs(y_true - y_pred) /
                                 np.clip(np.abs(y_true) + np.abs(y_pred), 1e-12, None)),
        "WAPE_%": 100 * np.abs(y_true - y_pred).sum() / np.abs(y_true).sum(),
        "MedAE": float(np.median(np.abs(y_true - y_pred))),
    }
    if (y_true >= 0).all() and (y_pred >= 0).all():
        out["RMSLE"] = np.sqrt(mean_squared_log_error(y_true, y_pred))
    if n_features:
        n = len(y_true)
        out["adj_R2"] = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return out


def mase(y_true, y_pred, y_train, m: int = 1) -> float:
    """m = chu kỳ mùa vụ (1 = naive, 7 = seasonal naive theo tuần)."""
    y_train = np.asarray(y_train)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)
```

### Phân tích phần dư (residual analysis)

```python
def residual_plots(y_true, y_pred):
    res = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(y_pred, res, s=5, alpha=0.4); axes[0].axhline(0, color="r")
    axes[0].set(xlabel="Predicted", ylabel="Residual", title="Residual vs Predicted")   # hình phễu => heteroscedasticity
    axes[1].scatter(y_true, y_pred, s=5, alpha=0.4)
    lim = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    axes[1].plot(lim, lim, "r--"); axes[1].set(xlabel="Actual", ylabel="Predicted", title="Actual vs Predicted")
    from scipy import stats
    stats.probplot(res, dist="norm", plot=axes[2]); axes[2].set_title("Q-Q plot residual")
    plt.tight_layout()
```

## 9.8. Metric phân cụm

```python
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

Xc = StandardScaler().fit_transform(
    X_train[["tenure_months", "monthly_charges", "support_calls"]].fillna(0))
for k in range(2, 9):
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xc)
    print(k,
          "silhouette↑", round(silhouette_score(Xc, labels, sample_size=5000, random_state=0), 3),
          "DB↓", round(davies_bouldin_score(Xc, labels), 3),
          "CH↑", round(calinski_harabasz_score(Xc, labels), 1))
# Có nhãn thật: adjusted_rand_score(y, labels), normalized_mutual_info_score(y, labels)
```

## 9.9. Metric truy hồi & xếp hạng (Search, RAG, Recommender)

```python
from sklearn.metrics import ndcg_score


def recall_at_k(relevant: set, ranked: list, k: int) -> float:
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def precision_at_k_list(relevant: set, ranked: list, k: int) -> float:
    return len(relevant & set(ranked[:k])) / k


def average_precision(relevant: set, ranked: list, k: int | None = None) -> float:
    ranked = ranked[:k] if k else ranked
    hits, score = 0, 0.0
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), len(ranked)) if relevant else 0.0


def reciprocal_rank(relevant: set, ranked: list) -> float:
    return next((1 / i for i, d in enumerate(ranked, 1) if d in relevant), 0.0)


def f_beta_sets(relevant: set, retrieved: set, beta: float = 2.0) -> float:
    """F2: ưu tiên recall gấp beta^2 = 4 lần precision — phù hợp truy hồi văn bản pháp luật."""
    tp = len(relevant & retrieved)
    if tp == 0:
        return 0.0
    p, r = tp / len(retrieved), tp / len(relevant)
    return (1 + beta**2) * p * r / (beta**2 * p + r)


# NDCG với độ liên quan phân cấp (graded relevance)
true_rel = np.array([[3, 2, 0, 0, 1]])       # độ liên quan thật của 5 tài liệu
scores = np.array([[0.9, 0.8, 0.7, 0.1, 0.6]])  # điểm mô hình
print("NDCG@3 =", ndcg_score(true_rel, scores, k=3))
# MAP@K, MRR = trung bình average_precision / reciprocal_rank qua tất cả truy vấn
```

## 9.10. So sánh mô hình có ý nghĩa thống kê

Mô hình B có PR-AUC 0.612 so với A là 0.605 — B có thực sự tốt hơn?

```python
from statsmodels.stats.contingency_tables import mcnemar


def paired_bootstrap(y_true, p_a, p_b, metric=average_precision_score, n_boot=2000, seed=0):
    """CI cho chênh lệch metric(B) - metric(A) trên CÙNG tập test (ghép cặp)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(metric(y_true[idx], p_b[idx]) - metric(y_true[idx], p_a[idx]))
    diffs = np.array(diffs)
    return {"mean_diff": diffs.mean(), "ci95": np.percentile(diffs, [2.5, 97.5]),
            "p_b_better": (diffs > 0).mean()}


def mcnemar_test(y_true, pred_a, pred_b):
    """So sánh 2 bộ phân loại (nhãn cứng) trên cùng tập test."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    table = [[np.sum(a_ok & b_ok), np.sum(a_ok & ~b_ok)],
             [np.sum(~a_ok & b_ok), np.sum(~a_ok & ~b_ok)]]
    return mcnemar(table, exact=False, correction=True).pvalue


def corrected_resampled_ttest(scores_a, scores_b, n_train: int, n_test: int):
    """Nadeau & Bengio (2003): t-test hiệu chỉnh cho điểm CV (các fold không độc lập)."""
    from scipy import stats
    d = np.asarray(scores_b) - np.asarray(scores_a)
    k = len(d)
    var = d.var(ddof=1) * (1 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    return t, 2 * stats.t.sf(abs(t), df=k - 1)
```

Nếu khoảng tin cậy của chênh lệch chứa 0 → **chưa đủ bằng chứng** B tốt hơn → giữ mô hình đơn giản hơn.

## 9.11. Phân tích lỗi & đánh giá theo phân khúc (Slice analysis)

```python
def slice_report(X: pd.DataFrame, y_true, proba, slice_col: str, threshold: float) -> pd.DataFrame:
    d = X[[slice_col]].copy()
    d["y"], d["p"] = np.asarray(y_true), proba
    d["pred"] = (d["p"] >= threshold).astype(int)
    rows = []
    for key, g in d.groupby(slice_col, observed=True):
        if g["y"].nunique() < 2:
            continue
        rows.append({slice_col: key, "n": len(g), "pos_rate": g["y"].mean(),
                     "roc_auc": roc_auc_score(g["y"], g["p"]),
                     "pr_auc": average_precision_score(g["y"], g["p"]),
                     "recall": recall_score(g["y"], g["pred"]),
                     "precision": precision_score(g["y"], g["pred"], zero_division=0)})
    return pd.DataFrame(rows).sort_values("roc_auc")


print(slice_report(X_valid, y_valid, proba_valid, "contract", t_profit).round(3))

# Xem các lỗi tự tin nhất (FN có điểm thấp nhất, FP có điểm cao nhất) để tìm pattern
errors = X_valid.assign(y=y_valid.values, p=proba_valid)
worst_fn = errors[(errors.y == 1)].nsmallest(20, "p")
worst_fp = errors[(errors.y == 0)].nlargest(20, "p")
```

## 9.12. Báo cáo đánh giá cuối cùng trên tập test

```python
def final_evaluation(model, X_test, y_test, threshold: float, n_boot: int = 1000) -> pd.DataFrame:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    rng = np.random.default_rng(0)
    y = np.asarray(y_test)
    metrics = {"roc_auc": lambda yy, pp, dd: roc_auc_score(yy, pp),
               "pr_auc": lambda yy, pp, dd: average_precision_score(yy, pp),
               "recall": lambda yy, pp, dd: recall_score(yy, dd),
               "precision": lambda yy, pp, dd: precision_score(yy, dd, zero_division=0),
               "f2": lambda yy, pp, dd: fbeta_score(yy, dd, beta=2),
               "brier": lambda yy, pp, dd: brier_score_loss(yy, pp)}
    rows = []
    for name, fn in metrics.items():
        point = fn(y, proba, pred)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                boots.append(fn(y[i], proba[i], pred[i]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"metric": name, "value": point, "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows).set_index("metric").round(4)
```

> **Checklist Chương 9**
> - [ ] Metric chính gắn với mục tiêu kinh doanh; có metric ràng buộc.
> - [ ] Không dùng Accuracy làm metric chính khi dữ liệu mất cân bằng.
> - [ ] Confusion matrix được phân tích ở ngưỡng thực sự sẽ deploy.
> - [ ] Ngưỡng chọn trên validation/OOF theo chi phí hoặc năng lực vận hành.
> - [ ] Đã kiểm tra calibration nếu xác suất được dùng trực tiếp.
> - [ ] Kết quả test có khoảng tin cậy; so sánh mô hình bằng kiểm định ghép cặp.
> - [ ] Có slice analysis theo các phân khúc quan trọng.
