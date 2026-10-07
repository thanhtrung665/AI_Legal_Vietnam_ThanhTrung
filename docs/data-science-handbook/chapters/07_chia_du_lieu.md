# CHƯƠNG 7. CHIA DỮ LIỆU & CHIẾN LƯỢC VALIDATION

> *"Your validation strategy matters more than your model."*

**Mục tiêu chương:** ước lượng **trung thực** hiệu năng của mô hình trên dữ liệu tương lai mà nó chưa từng thấy. Muốn vậy, cách chia dữ liệu phải **mô phỏng đúng cách mô hình được dùng trong production**: dự đoán cho *thời điểm sau*, cho *khách hàng mới*, với *thông tin có sẵn lúc đó*.

## 7.1. Ba tập dữ liệu và nguyên tắc sử dụng

| Tập | Vai trò | Dùng để | Tỷ lệ điển hình |
|---|---|---|---|
| **Train** | Học tham số | `fit` | 60–80% |
| **Validation** (hoặc CV trên train) | Ra quyết định: chọn mô hình, siêu tham số, ngưỡng, early stopping, feature | So sánh | 10–20% |
| **Test (hold-out)** | Ước lượng cuối cùng, **một lần** | Báo cáo | 10–20% |

> **Quy tắc vàng.** Tập test bị "đốt" ngay khi bạn dùng nó để ra **bất kỳ** quyết định nào. Lặp lại vòng "train → xem test → chỉnh → xem test" biến test thành validation và kết quả báo cáo bị lạc quan. Andrew Ng gọi đây là *overfitting the dev set*. Khi điều đó xảy ra, cần một tập dev mới.

### Thiết lập

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GroupKFold, KFold, RepeatedStratifiedKFold, StratifiedGroupKFold,
                                     StratifiedKFold, TimeSeriesSplit, cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.pipeline import make_pipeline

from churn.data.split import assert_no_leakage, purged_time_splits, time_split
from churn.data.synthetic import make_churn_data
from churn.features.build import RAW_FEATURES, build_preprocessor
from churn.features.clean import clean_churn

pd.set_option("display.width", 160)
df = clean_churn(make_churn_data(n=20_000, seed=42))
X, y = df[RAW_FEATURES], df["churn"]
rng = np.random.default_rng(7)
```

## 7.2. Bảng chọn chiến lược chia

| Tình huống | Chiến lược | scikit-learn |
|---|---|---|
| i.i.d., cân bằng | Random split / K-Fold | `train_test_split`, `KFold(shuffle=True)` |
| Phân loại mất cân bằng | **Stratified** | `StratifiedKFold`, `train_test_split(stratify=y)` |
| Nhiều bản ghi / một thực thể | **Group**: một nhóm chỉ thuộc một tập | `GroupKFold`, **`StratifiedGroupKFold`**, `GroupShuffleSplit`, `LeaveOneGroupOut` |
| Dự đoán tương lai | **Out-of-time** | Chia theo ngày; `TimeSeriesSplit` |
| Nhãn có cửa sổ tương lai (chồng lấn) | Time split + **purging + embargo** | Tự viết (`churn.data.split.purged_time_splits`) |
| Dữ liệu ít | **Repeated** stratified K-Fold | `RepeatedStratifiedKFold` |
| Vừa tune vừa ước lượng không bias | **Nested CV** | `GridSearchCV` bên trong `cross_val_score` |
| Có sẵn tập validation cố định | Predefined split | `PredefinedSplit` |
| Dữ liệu không gian | Spatial/block CV | `GroupKFold` theo ô lưới |

**[Docs]** *A note on shuffling*: nếu thứ tự dữ liệu không ngẫu nhiên (ví dụ dữ liệu được sắp theo nhãn), cần shuffle để CV có ý nghĩa. Ngược lại, nếu các mẫu **không i.i.d.**, ví dụ bài báo xếp theo thời gian xuất bản, thì shuffle lại làm **điểm validation bị thổi phồng**, vì mô hình được kiểm tra trên mẫu "giống" mẫu huấn luyện một cách giả tạo.

## 7.3. Hold-out ba tập (stratified)

```python
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
for name, t in [("train", y_tr), ("valid", y_va), ("test", y_te)]:
    print(f"{name:5s} n={len(t):6d} churn_rate={t.mean():.4f}")
```

### Tập validation/test cần lớn bao nhiêu?

Sai số chuẩn của AUC (công thức Hanley & McNeil, 1982) cho biết **độ chính xác của chính phép đo**:

$$SE(AUC) = \sqrt{\frac{A(1-A) + (n_1 - 1)(Q_1 - A^2) + (n_0 - 1)(Q_2 - A^2)}{n_1 n_0}},\quad Q_1 = \frac{A}{2-A},\; Q_2 = \frac{2A^2}{1+A}$$

```python
def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    return np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg))


for n in [500, 2_000, 10_000]:
    se = auc_standard_error(0.75, int(n * 0.165), int(n * 0.835))
    print(f"n_test={n:6d}: AUC 0.75 ± {1.96 * se:.3f} (95% CI)")
# Với 500 mẫu, hai mô hình chênh 0.02 AUC là KHÔNG phân biệt được
```

## 7.4. Out-of-time: chuẩn cho hầu hết bài toán kinh doanh

```python
train_df, valid_df, test_df = time_split(df, "signup_date", "2023-10-01", "2024-01-01", gap_days=0)
assert_no_leakage(train_df, test_df, id_col="customer_id", date_col="signup_date")
print({k: (len(v), v["signup_date"].min().date(), v["signup_date"].max().date())
       for k, v in [("train", train_df), ("valid", valid_df), ("test", test_df)]})
```

### Vì sao random split có thể đánh lừa: mô phỏng concept drift

Khi quan hệ X → y **thay đổi theo thời gian** (concept drift), random split trộn tương lai vào train và cho điểm lạc quan. Điểm out-of-time mới phản ánh thực tế.

```python
n = 12_000
t = np.sort(rng.uniform(0, 1, n))                          # thời gian chuẩn hóa [0, 1]
x1, x2 = rng.normal(size=n), rng.normal(size=n)
beta1 = 1.5 - 2.5 * t                                      # tác động của x1 đảo chiều theo thời gian
p = 1 / (1 + np.exp(-(-1 + beta1 * x1 + 0.8 * x2)))
drift = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": (rng.random(n) < p).astype(int)})

model = LogisticRegression()
random_cv = cross_val_score(model, drift[["x1", "x2"]], drift["y"],
                            cv=KFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()
past, future = drift[drift.t < 0.8], drift[drift.t >= 0.8]
oot = roc_auc_score(future["y"], model.fit(past[["x1", "x2"]], past["y"]).predict_proba(future[["x1", "x2"]])[:, 1])
print(f"Random 5-fold CV AUC = {random_cv:.3f} | Out-of-time AUC = {oot:.3f}")
```

**[Kinh nghiệm]** Đánh giá production nên vừa **out-of-time** (giai đoạn sau) vừa **out-of-sample** (khách hàng mới). Báo cáo hiệu năng **theo từng tháng** của giai đoạn test để thấy xu hướng suy giảm.

## 7.5. Cross-validation đúng cách

### Hồ sơ các bộ chia (minh họa chỉ số)

```python
toy = np.arange(12)
toy_y = np.array([0] * 8 + [1] * 4)
toy_groups = np.repeat(["A", "B", "C", "D", "E", "F"], 2)
splitters = {
    "KFold(3)": KFold(3),
    "StratifiedKFold(3)": StratifiedKFold(3),
    "GroupKFold(3)": GroupKFold(3),
    "TimeSeriesSplit(3)": TimeSeriesSplit(3),
    "TimeSeriesSplit(3,gap=1)": TimeSeriesSplit(3, gap=1),
}
for name, sp in splitters.items():
    folds = [f"train={tr.tolist()} val={va.tolist()}" for tr, va in sp.split(toy, toy_y, groups=toy_groups)]
    print(f"{name}:\n   " + "\n   ".join(folds))
```

### `cross_validate`: nhiều metric, điểm train, chỉ số fold

```python
pipe = make_pipeline(build_preprocessor(), LogisticRegression(max_iter=3000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)       # số nguyên: cùng fold cho mọi mô hình
res = cross_validate(pipe, X_tr, y_tr, cv=cv, n_jobs=-1, return_train_score=True, return_indices=True,
                     scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "neg_log_loss": "neg_log_loss"})
summary = pd.DataFrame({k: v for k, v in res.items() if k.startswith(("train_", "test_"))})
print(summary.agg(["mean", "std"]).T.round(4))
# train >> test: overfitting | std lớn: mô hình không ổn định hoặc dữ liệu ít
```

**[Docs]** *Note on inappropriate usage of `cross_val_predict`*: kết quả của `cross_val_predict` có thể khác `cross_val_score`, vì nó trả về dự đoán của **nhiều mô hình khác nhau** gộp chung lại. Do đó nó **không phải** thước đo sai số tổng quát hóa phù hợp. Hãy dùng nó cho: (1) out-of-fold predictions để **chọn ngưỡng / calibrate / stacking**, (2) trực quan hóa.

```python
from sklearn.model_selection import cross_val_predict

oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
print("OOF predictions:", oof.shape, "| dùng cho threshold tuning & calibration (Chương 9)")
```

### Bao nhiêu fold? Lặp lại CV

- K nhỏ (5): mỗi mô hình huấn luyện trên ít dữ liệu hơn, nên ước lượng hơi **bi quan** (bias), nhưng nhanh.
- K lớn (10, LOOCV): ít bias, nhưng **phương sai cao** (các tập train gần trùng nhau) và chậm.
- Thực hành tốt: K = 5 hoặc 10. Với dữ liệu nhỏ, dùng **lặp lại** (`RepeatedStratifiedKFold`) để giảm phương sai của ước lượng.

```python
small = X_tr.iloc[:1500], y_tr.iloc[:1500]
for name, splitter in [("5-fold x1", StratifiedKFold(5, shuffle=True, random_state=0)),
                       ("5-fold x5", RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0))]:
    s = cross_val_score(pipe, *small, cv=splitter, scoring="roc_auc", n_jobs=-1)
    print(f"{name:10s}: {s.mean():.4f} ± {s.std():.4f}  (n_fits={len(s)})")
```

> **Lưu ý thống kê.** Các fold CV **không độc lập** (tập train chồng lấn), nên `std` qua các fold **đánh giá thấp** độ bất định thật. So sánh hai mô hình bằng t-test thông thường trên điểm fold là sai. Cần dùng *corrected resampled t-test* (Nadeau & Bengio, 2003), xem Chương 9.10.

### Permutation test score: mô hình có học được gì không?

**[Docs]** `permutation_test_score` xáo trộn nhãn *n* lần để tạo phân phối null. H₀ là "mô hình không khai thác được phụ thuộc nào giữa feature và target". Hữu ích khi dữ liệu nhỏ và muốn chắc chắn điểm số không do may mắn.

```python
from sklearn.model_selection import permutation_test_score

score, perm_scores, pvalue = permutation_test_score(
    make_pipeline(build_preprocessor(), LogisticRegression(max_iter=2000)), X_tr.iloc[:800], y_tr.iloc[:800],
    scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=0), n_permutations=50,
    random_state=0, n_jobs=-1)
print(f"AUC thật={score:.3f} | AUC khi xáo nhãn={perm_scores.mean():.3f}±{perm_scores.std():.3f} | p={pvalue:.3f}")
```

## 7.6. Group K-Fold: tránh rò rỉ theo thực thể

Khi một khách hàng có nhiều snapshot (mỗi tháng một dòng), random split để cùng khách hàng xuất hiện ở cả train và validation. Mô hình "nhớ" khách hàng thay vì học quy luật tổng quát.

```python
# Dữ liệu panel: 1 500 khách × 8 tháng; mỗi khách có "độ trung thành" ẩn ổn định theo thời gian
n_cust, n_months = 1_500, 8
cust = np.repeat(np.arange(n_cust), n_months)
latent = np.repeat(rng.normal(0, 2.5, n_cust), n_months)          # đặc tính riêng, ổn định của từng khách
fingerprint = np.repeat(rng.uniform(0, 1, n_cust), n_months)       # feature định danh vô nghĩa (vd: mã vùng chi tiết)
usage = rng.normal(0, 1, n_cust * n_months)
yp = (rng.random(n_cust * n_months) < 1 / (1 + np.exp(-(-1.5 + latent + 1.0 * usage)))).astype(int)
panel = pd.DataFrame({"fingerprint": fingerprint, "usage": usage, "y": yp, "customer": cust})

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
Xp, yp_ = panel[["fingerprint", "usage"]], panel["y"]
random_auc = cross_val_score(rf, Xp, yp_, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()
group_auc = cross_val_score(rf, Xp, yp_, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                            groups=panel["customer"], scoring="roc_auc").mean()
print(f"Random K-Fold AUC = {random_auc:.3f}  <- ảo: mô hình nhận ra khách qua 'fingerprint'")
print(f"Group  K-Fold AUC = {group_auc:.3f}  <- trung thực với khách hàng mới")
```

## 7.7. Chuỗi thời gian: TimeSeriesSplit, gap, purging, embargo

**[Docs]** `TimeSeriesSplit` tạo các fold **expanding window**: train luôn ở trước validation. Các tham số quan trọng:

- `gap`: số **mẫu** bị loại giữa train và validation (không phải số ngày; dữ liệu phải được sắp theo thời gian).
- `max_train_size`: giới hạn cửa sổ train (**sliding window**) khi dữ liệu cũ không còn đại diện.
- `test_size`: kích thước mỗi fold validation.

```text
Expanding window (mặc định)                Sliding window (max_train_size)
Fold 1: [TRAIN]──gap──[VAL]                Fold 1: [TRAIN]──gap──[VAL]
Fold 2: [TRAIN TRAIN]──gap──[VAL]          Fold 2:    [TRAIN]──gap──[VAL]
Fold 3: [TRAIN TRAIN TRAIN]──gap──[VAL]    Fold 3:       [TRAIN]──gap──[VAL]
```

```python
ordered = df.sort_values("signup_date").reset_index(drop=True)
tscv = TimeSeriesSplit(n_splits=4, gap=300, test_size=2_000)
rows = []
for k, (tr, va) in enumerate(tscv.split(ordered)):
    m = make_pipeline(build_preprocessor(), LogisticRegression(max_iter=3000)).fit(
        ordered.loc[tr, RAW_FEATURES], ordered.loc[tr, "churn"])
    auc = roc_auc_score(ordered.loc[va, "churn"], m.predict_proba(ordered.loc[va, RAW_FEATURES])[:, 1])
    rows.append({"fold": k, "train_end": ordered.loc[tr[-1], "signup_date"].date(),
                 "valid": f"{ordered.loc[va[0], 'signup_date'].date()} → {ordered.loc[va[-1], 'signup_date'].date()}",
                 "n_train": len(tr), "auc": round(auc, 4)})
print(pd.DataFrame(rows).to_string(index=False))     # backtest: hiệu năng qua từng giai đoạn
```

### Purged K-Fold với Embargo (nhãn chồng lấn)

Khi nhãn tại thời điểm t phụ thuộc dữ liệu trong khoảng [t, t+h] (churn trong 30 ngày tới, lợi nhuận 5 ngày tới), mẫu train **gần ranh giới** có nhãn "nhìn thấy" giai đoạn validation. López de Prado (2018, ch. 7) đề xuất **purge** (loại mẫu train có cửa sổ nhãn chồng lên validation) và **embargo** (loại thêm một khoảng sau validation):

```python
dates = df["signup_date"].reset_index(drop=True)
for k, (tr, va) in enumerate(purged_time_splits(dates, n_splits=5, horizon=pd.Timedelta(days=30),
                                                 embargo=pd.Timedelta(days=7))):
    removed = len(dates) - len(tr) - len(va)
    print(f"fold {k}: train={len(tr):5d} valid={len(va):5d} bị purge/embargo={removed:4d}")
```

## 7.8. Nested CV: ước lượng không bias khi có tuning

Dùng cùng một CV để vừa **chọn siêu tham số** vừa **báo cáo điểm** cho kết quả lạc quan, vì ta đã chọn cấu hình "may mắn" nhất trên chính các fold đó. **[Docs]** Ví dụ chính thức *Nested versus non-nested cross-validation* (bộ Iris, SVC) cho thấy điểm non-nested cao hơn có hệ thống.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

Xb, yb = load_breast_cancer(return_X_y=True)
idx = np.random.default_rng(0).permutation(len(yb))[:120]           # dữ liệu nhỏ: bias lạc quan lộ rõ
Xb, yb = Xb[idx], yb[idx]
grid = {"svc__C": [0.01, 0.1, 1, 10, 100], "svc__gamma": [1e-4, 1e-3, 1e-2, 0.1, 1]}
non_nested, nested = [], []
for trial in range(5):
    inner = StratifiedKFold(4, shuffle=True, random_state=trial)
    outer = StratifiedKFold(4, shuffle=True, random_state=100 + trial)
    search = GridSearchCV(make_pipeline(StandardScaler(), SVC()), grid, cv=inner, scoring="accuracy")
    search.fit(Xb, yb)
    non_nested.append(search.best_score_)                          # chọn và báo cáo trên cùng fold
    nested.append(cross_val_score(search, Xb, yb, cv=outer, scoring="accuracy").mean())
diff = np.array(non_nested) - np.array(nested)
print(f"Non-nested={np.mean(non_nested):.4f} | Nested={np.mean(nested):.4f} | chênh lệch lạc quan={diff.mean():+.4f}")
```

Chênh lệch tăng khi **dữ liệu ít** và **lưới tìm kiếm lớn** (càng nhiều cấu hình, càng dễ gặp cấu hình "may mắn"). Trong thực tế: dùng nested CV để **báo cáo** hiệu năng của *quy trình* (bao gồm tuning). Dùng `search.best_estimator_` refit trên toàn bộ train làm mô hình cuối, rồi kiểm tra một lần trên test out-of-time.

## 7.9. Danh mục data leakage

Kapoor & Narayanan (2023), *Leakage and the reproducibility crisis in ML-based science*, tổng hợp 294 bài báo bị ảnh hưởng bởi leakage. Có thể phân loại như sau:

| Loại | Ví dụ | Phòng tránh |
|---|---|---|
| **Không tách train/test** | Feature selection, impute, scale trên toàn bộ dữ liệu | `Pipeline` + CV (Chương 5) |
| **Tiền xử lý dùng thông tin test** | Target encoding không cross-fitting | `TargetEncoder.fit_transform` |
| **Feature không hợp lệ** (target leakage) | Feature chỉ có sau sự kiện (refund, lý do hủy) | Kiểm tra thời điểm sinh feature; AUC đơn biến |
| **Phân phối test không khớp phạm vi áp dụng** | Test ngẫu nhiên khi production là tương lai | Out-of-time, out-of-sample |
| **Phụ thuộc thời gian** | Random split dữ liệu chuỗi thời gian; dùng dữ liệu tương lai | Time split, point-in-time join |
| **Phụ thuộc giữa train và test** | Cùng khách hàng/bệnh nhân ở cả hai tập | Group split |
| **Trùng lặp** | Bản ghi trùng hoặc gần trùng giữa train/test | Dedup trước khi chia (cả near-duplicate) |
| **Tuning trên test** | Chọn mô hình theo điểm test | Validation riêng / nested CV |

```python
def leakage_audit(train: pd.DataFrame, test: pd.DataFrame, id_col: str, date_col: str,
                  feature_cols: list[str]) -> dict:
    """Các kiểm tra tự động nên chạy trong CI cho mọi lần chia dữ liệu."""
    dup = train[feature_cols].merge(test[feature_cols], how="inner").shape[0]
    return {
        "id_overlap": len(set(train[id_col]) & set(test[id_col])),
        "time_overlap": bool(train[date_col].max() >= test[date_col].min()),
        "exact_feature_duplicates": int(dup),
        "test_rate_vs_train": round(test["churn"].mean() / train["churn"].mean(), 3),
    }


print(leakage_audit(train_df, test_df, "customer_id", "signup_date",
                    ["age", "tenure_months", "monthly_charges", "contract", "support_calls"]))
```

> **Checklist Chương 7**
> - [ ] Cách chia mô phỏng đúng kịch bản production (thời gian, thực thể mới, thông tin có sẵn).
> - [ ] Stratify cho phân loại mất cân bằng; group split khi một thực thể có nhiều dòng.
> - [ ] Dữ liệu thời gian: out-of-time / `TimeSeriesSplit` (+ gap, purge, embargo khi nhãn có cửa sổ tương lai).
> - [ ] `random_state` là số nguyên cho splitter; mọi mô hình được so sánh trên cùng fold.
> - [ ] Báo cáo mean ± std (hoặc CI) qua fold; đủ cỡ mẫu validation để phân biệt các mô hình.
> - [ ] Tuning dùng validation/inner CV; báo cáo bằng nested CV hoặc test out-of-time một lần.
> - [ ] `cross_val_predict` chỉ dùng cho OOF (threshold, calibration, stacking), không để báo cáo điểm.
> - [ ] Có kiểm tra leakage tự động (ID, thời gian, trùng lặp) trong CI.

### Tài liệu tham khảo Chương 7

- scikit-learn User Guide: *Cross-validation: evaluating estimator performance* (iterators, `cross_validate`, `cross_val_predict` warning, *A note on shuffling*, *Permutation test score*); *Tuning the hyper-parameters of an estimator*; *Common pitfalls*.
- scikit-learn Examples: *Nested versus non-nested cross-validation*; *Visualizing cross-validation behavior in scikit-learn*.
- Hanley, J. A. & McNeil, B. J. (1982). *The meaning and use of the area under a ROC curve.* Radiology 143(1).
- Nadeau, C. & Bengio, Y. (2003). *Inference for the Generalization Error.* Machine Learning 52.
- Cawley, G. C. & Talbot, N. L. C. (2010). *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.* JMLR 11.
- Kaufman, S. et al. (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM TKDD 6(4).
- Kapoor, S. & Narayanan, A. (2023). *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns 4(9).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 7. Wiley.
- Ng, A. *Machine Learning Yearning*, ch. 5–12 (dev/test sets). · Google ML Crash Course: *Datasets, generalization, and overfitting*.
