# CHƯƠNG 7. CHIA DỮ LIỆU & CHIẾN LƯỢC VALIDATION

> *"Chiến lược validation của bạn quan trọng hơn mô hình của bạn."* — Mọi Kaggle Grandmaster.

Mục tiêu duy nhất của việc chia dữ liệu: **ước lượng trung thực hiệu năng của mô hình trên dữ liệu tương lai mà nó chưa từng thấy**. Vì vậy cách chia phải **mô phỏng đúng cách mô hình được dùng trong production**.

## 7.1. Ba tập dữ liệu và vai trò

| Tập | Vai trò | Được dùng để | Tỷ lệ thường gặp |
|---|---|---|---|
| **Train** | Học tham số mô hình | `fit` | 60–80% |
| **Validation** | Chọn mô hình, siêu tham số, ngưỡng, early stopping | Ra quyết định | 10–20% (hoặc dùng CV) |
| **Test (hold-out)** | Ước lượng hiệu năng cuối cùng | **Chỉ đánh giá 1 lần** | 10–20% |

> **Quy tắc vàng:** Tập test bị "đốt" ngay khi bạn dùng nó để ra bất kỳ quyết định nào. Nếu bạn lặp lại "train → xem test → chỉnh → xem test", test đã trở thành validation.

## 7.2. Bảng chọn chiến lược chia

| Tình huống | Chiến lược | sklearn |
|---|---|---|
| Dữ liệu i.i.d., cân bằng | Random split / KFold | `train_test_split`, `KFold` |
| Phân loại mất cân bằng | **Stratified** | `StratifiedKFold`, `stratify=y` |
| Nhiều bản ghi / một thực thể (khách hàng, bệnh nhân) | **Group** — một nhóm chỉ ở 1 tập | `GroupKFold`, `StratifiedGroupKFold`, `GroupShuffleSplit` |
| Dữ liệu có thời gian, dự đoán tương lai | **Time-based** (out-of-time) | `TimeSeriesSplit`, split theo ngày |
| Chuỗi thời gian có độ trễ nhãn | Time split + **gap / purging / embargo** | `TimeSeriesSplit(gap=...)` |
| Dữ liệu ít (< vài nghìn mẫu) | Repeated (Stratified) KFold | `RepeatedStratifiedKFold` |
| Vừa tune vừa ước lượng không bias | **Nested CV** | `GridSearchCV` lồng trong `cross_val_score` |
| Dữ liệu không gian (địa lý) | Spatial / block CV | `GroupKFold` theo ô lưới |

## 7.3. Hold-out split cơ bản (stratified)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn.data.synthetic import make_churn_data
from churn.features.clean import clean_churn

df = clean_churn(make_churn_data())
X, y = df.drop(columns=["churn"]), df["churn"]

# Chia 2 bước: 70% train / 15% valid / 15% test
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

for name, t in [("train", y_train), ("valid", y_valid), ("test", y_test)]:
    print(f"{name:5s} n={len(t):6d} churn_rate={t.mean():.4f}")
```

## 7.4. Out-of-time split — chuẩn cho hầu hết bài toán kinh doanh

```python
def time_split(df: pd.DataFrame, date_col: str, train_end: str, valid_end: str,
               gap_days: int = 0):
    """train: < train_end | (gap) | valid: [train_end+gap, valid_end) | test: >= valid_end + gap"""
    d = df[date_col]
    gap = pd.Timedelta(days=gap_days)
    train = df[d < pd.Timestamp(train_end)]
    valid = df[(d >= pd.Timestamp(train_end) + gap) & (d < pd.Timestamp(valid_end))]
    test = df[d >= pd.Timestamp(valid_end) + gap]
    return train, valid, test


train_df, valid_df, test_df = time_split(df, "signup_date", "2023-12-01", "2024-03-01", gap_days=30)
```

Trong production, khuyến nghị đánh giá thêm **"out-of-time" + "out-of-sample"**: test là khách hàng mới **và** ở giai đoạn sau.

## 7.5. Cross-Validation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GroupKFold, RepeatedStratifiedKFold, StratifiedGroupKFold,
                                     StratifiedKFold, TimeSeriesSplit, cross_validate)
from sklearn.pipeline import make_pipeline

from churn.features.build import build_preprocessor

pipe = make_pipeline(build_preprocessor(), LogisticRegression(max_iter=2000))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    pipe, X_train, y_train, cv=cv,
    scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"},
    return_train_score=True, n_jobs=-1,
)
res = pd.DataFrame(scores)
print(res.agg(["mean", "std"]).T.round(4))
# train_score >> test_score  => overfitting ; std lớn => mô hình không ổn định / dữ liệu ít
```

### Group K-Fold — tránh rò rỉ theo thực thể

Nếu một khách hàng có nhiều snapshot (mỗi tháng 1 dòng), random split để cùng khách hàng xuất hiện ở cả train và valid → mô hình "nhớ" khách hàng → metric ảo.

```python
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
groups = X_train["customer_id"]
for fold, (tr, va) in enumerate(sgkf.split(X_train, y_train, groups=groups)):
    assert set(groups.iloc[tr]).isdisjoint(groups.iloc[va])
```

### Time Series Split (expanding / sliding window)

```python
X_sorted = X_train.sort_values("signup_date")
y_sorted = y_train.loc[X_sorted.index]

# Lưu ý: gap tính theo SỐ MẪU (dòng), không phải số ngày -> dữ liệu phải được sắp xếp theo thời gian
tscv = TimeSeriesSplit(n_splits=5, gap=500, test_size=None, max_train_size=None)
# max_train_size=N -> sliding window (khi dữ liệu cũ không còn đại diện)
for fold, (tr, va) in enumerate(tscv.split(X_sorted)):
    print(fold, X_sorted.iloc[tr]["signup_date"].max().date(), "->",
          X_sorted.iloc[va]["signup_date"].min().date())
```

```text
Expanding window:                     Sliding window:
Fold 1: [TRAIN][gap][VAL]             Fold 1: [TRAIN][gap][VAL]
Fold 2: [TRAIN  TRAIN][gap][VAL]      Fold 2:    [TRAIN][gap][VAL]
Fold 3: [TRAIN  TRAIN  TRAIN][gap][VAL]  Fold 3:       [TRAIN][gap][VAL]
```

### Purged K-Fold với Embargo (tài chính, nhãn chồng lấn)

Khi nhãn tại thời điểm t phụ thuộc dữ liệu trong [t, t+h] (ví dụ lợi nhuận 5 ngày), mẫu train gần ranh giới validation có nhãn "chồng lấn" với validation → cần loại bỏ (purge) và thêm vùng cấm (embargo) — xem *López de Prado, Advances in Financial ML, 2018*.

```python
from collections.abc import Iterator


def purged_time_splits(dates: pd.Series, n_splits: int = 5, horizon: pd.Timedelta = pd.Timedelta(days=30),
                       embargo: pd.Timedelta = pd.Timedelta(days=7)) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    dates = dates.reset_index(drop=True)
    order = np.argsort(dates.to_numpy())
    folds = np.array_split(order, n_splits)
    for va_idx in folds:
        va_start, va_end = dates.iloc[va_idx].min(), dates.iloc[va_idx].max()
        mask = ((dates + horizon) < va_start) | (dates > va_end + embargo)   # purge + embargo
        tr_idx = np.where(mask.to_numpy())[0]
        yield tr_idx, va_idx
```

## 7.6. Nested Cross-Validation — ước lượng không bias khi có tuning

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

inner = StratifiedKFold(5, shuffle=True, random_state=1)
outer = StratifiedKFold(5, shuffle=True, random_state=2)

search = GridSearchCV(
    pipe, param_grid={"logisticregression__C": [0.01, 0.1, 1, 10]},
    cv=inner, scoring="average_precision", n_jobs=-1,
)
nested_scores = cross_val_score(search, X_train, y_train, cv=outer, scoring="average_precision")
print(f"Nested CV PR-AUC: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
```

Chọn tham số trên chính tập dùng để báo cáo điểm → điểm bị **lạc quan (optimistic bias)**. Nested CV tách hai vai trò này.

## 7.7. Out-of-Fold (OOF) predictions — nền tảng cho stacking & threshold tuning

```python
from sklearn.model_selection import cross_val_predict

oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
# oof_proba: mỗi mẫu train được dự đoán bởi mô hình KHÔNG thấy nó -> dùng để
# (1) chọn ngưỡng, (2) calibrate, (3) huấn luyện meta-model khi stacking
```

## 7.8. Danh mục Data Leakage — kẻ thù số 1

| Loại leakage | Ví dụ | Cách phòng |
|---|---|---|
| **Target leakage** | Feature `refund_amount` chỉ có sau khi khách hủy | Kiểm tra thời điểm sinh của từng feature |
| **Train-test contamination** | Scale/impute/SMOTE/feature selection trên toàn bộ dữ liệu | Pipeline + CV |
| **Temporal leakage** | Random split cho dữ liệu thời gian; feature dùng dữ liệu tương lai | Time split, point-in-time join |
| **Group leakage** | Cùng khách hàng/bệnh nhân ở train và test | Group split |
| **Duplicate leakage** | Bản ghi gần trùng ở train và test | Dedup trước split (cả near-duplicate) |
| **Preprocessing leakage trong target encoding** | Mean encoding tính cả target của chính dòng đó | Cross-fitting, `TargetEncoder` |
| **Hyperparameter leakage** | Tune trên test | Validation riêng / nested CV |

### Test tự động chống leakage

```python
def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame, id_col: str, date_col: str | None = None):
    overlap = set(train[id_col]) & set(test[id_col])
    assert not overlap, f"{len(overlap)} ID xuất hiện ở cả train và test"
    if date_col:
        assert train[date_col].max() < test[date_col].min(), "Train chứa dữ liệu sau thời điểm test"
```

> **Checklist Chương 7**
> - [ ] Cách chia mô phỏng đúng kịch bản production (thời gian, thực thể mới).
> - [ ] Stratify cho phân loại mất cân bằng; group split khi có nhiều bản ghi/thực thể.
> - [ ] Tập test chỉ dùng 1 lần cuối cùng; mọi quyết định dựa trên validation/CV.
> - [ ] Có test tự động kiểm tra leakage (ID, thời gian, trùng lặp).
> - [ ] Báo cáo mean ± std qua các fold, không chỉ 1 con số.
