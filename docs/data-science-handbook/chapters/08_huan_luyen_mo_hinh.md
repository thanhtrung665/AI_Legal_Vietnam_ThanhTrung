# CHƯƠNG 8. HUẤN LUYỆN MÔ HÌNH (MODEL TRAINING)

**Mục tiêu chương:** chọn họ mô hình phù hợp, huấn luyện đúng cách (baseline, early stopping, regularization), tối ưu siêu tham số hiệu quả, chẩn đoán bias/variance, và đóng gói quá trình huấn luyện thành **một lệnh tái lập được, có tracking**.

## 8.1. Nền tảng lý thuyết

### Tối thiểu hóa rủi ro thực nghiệm (ERM)

Học có giám sát tìm hàm $f$ trong một lớp giả thuyết $\mathcal{F}$ sao cho tối thiểu **rủi ro thực nghiệm có regularization**:

$$\hat f = \arg\min_{f \in \mathcal{F}} \frac{1}{n}\sum_{i=1}^n L\big(y_i, f(x_i)\big) + \lambda\,\Omega(f)$$

Thứ ta thực sự quan tâm là **rủi ro kỳ vọng** trên phân phối thật $\mathbb{E}_{(x,y)\sim P}[L(y, f(x))]$. Khoảng cách giữa hai đại lượng này là **generalization gap**. Toàn bộ Chương 7 nhằm ước lượng đúng khoảng cách đó.

### Chọn hàm loss theo đại lượng cần dự đoán

**[Docs]** scikit-learn *Which scoring function should I use?* (dựa trên Gneiting, 2011): chọn **hàm loss nhất quán chặt (strictly consistent)** với đại lượng thống kê (functional) cần dự đoán. Nên dùng cùng hàm đó cho cả huấn luyện và đánh giá.

| Đại lượng cần dự đoán | Loss nhất quán | scikit-learn / LightGBM |
|---|---|---|
| Xác suất lớp (mean của Y nhị phân) | **Log loss**, Brier | `LogisticRegression`, `objective="binary"` |
| Mean của Y liên tục | **Squared error** | `loss="squared_error"`, `objective="regression"` |
| Median | **Absolute error** | `loss="absolute_error"`, `objective="l1"` |
| Quantile α | **Pinball loss** | `loss="quantile", quantile=α`, `objective="quantile"` |
| Mean của dữ liệu đếm ≥ 0 | Poisson deviance | `loss="poisson"`, `objective="poisson"` |
| Mean của Y dương lệch phải | Gamma / Tweedie deviance | `loss="gamma"`, `objective="tweedie"` |
| Xếp hạng | Pairwise/listwise (LambdaRank) | `LGBMRanker(objective="lambdarank")` |

### Phân rã bias–variance

Với loss bình phương: $\mathbb{E}[(y - \hat f(x))^2] = \underbrace{\sigma^2}_{\text{nhiễu}} + \underbrace{(\mathbb{E}\hat f(x) - f(x))^2}_{\text{bias}^2} + \underbrace{\operatorname{Var}(\hat f(x))}_{\text{variance}}$.

**[Course]** Andrew Ng (*Machine Learning Yearning*; *Deep Learning Specialization*, khóa 3) đưa ra quy trình chẩn đoán:

| Train error | Dev error | So với mức tối ưu (Bayes/con người) | Chẩn đoán | Hành động |
|---|---|---|---|---|
| Cao | Cao (≈ train) | Xa | **Avoidable bias** | Mô hình lớn hơn, thêm feature, giảm regularization, train lâu hơn |
| Thấp | Cao | Gần | **Variance** | Thêm dữ liệu, regularization, early stopping, giảm độ phức tạp, bớt feature |
| Thấp | Thấp, nhưng test/production cao | — | **Data mismatch** / leakage | Sửa cách chia (Chương 7), adversarial validation |

### Thiết lập

```python
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline

from churn.data.split import time_split
from churn.data.synthetic import make_churn_data
from churn.features.build import RAW_FEATURES, build_preprocessor
from churn.features.clean import clean_churn

pd.set_option("display.width", 160)
df = clean_churn(make_churn_data(n=20_000, seed=42))
train_df, valid_df, test_df = time_split(df, "signup_date", "2023-10-01", "2024-01-01")
X_train, y_train = train_df[RAW_FEATURES], train_df["churn"]
X_valid, y_valid = valid_df[RAW_FEATURES], valid_df["churn"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
print(X_train.shape, X_valid.shape, round(pos_weight, 2))
```

## 8.2. Baseline: luôn bắt đầu từ đây

```python
dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
p_dummy = dummy.predict_proba(X_valid)[:, 1]


def rule_based_score(X: pd.DataFrame) -> np.ndarray:
    """Heuristic CSKH đang dùng: hợp đồng tháng + nhiều khiếu nại + khách mới."""
    return ((X["contract"] == "month-to-month").astype(int) * 2 + (X["support_calls"] >= 3).astype(int)
            + (X["tenure_months"] < 6).astype(int)).to_numpy()


baselines = pd.DataFrame({
    "dummy(prior)": [roc_auc_score(y_valid, p_dummy), average_precision_score(y_valid, p_dummy)],
    "rule-based": [roc_auc_score(y_valid, rule_based_score(X_valid)),
                   average_precision_score(y_valid, rule_based_score(X_valid))],
}, index=["roc_auc", "pr_auc"]).T
print(baselines.round(4))   # PR-AUC của dummy = prevalence: sàn tuyệt đối
```

## 8.3. Các họ mô hình: cơ chế, giả định, tham số chính

| Họ | Cơ chế | Mạnh | Yếu | Tham số quan trọng |
|---|---|---|---|---|
| **Logistic / Linear** | Tổ hợp tuyến tính + link function | Nhanh, diễn giải tốt, xác suất calibrate tốt khi đúng đặc tả | Cần feature engineering cho phi tuyến | `C`, `l1_ratio`, `solver`, `class_weight` |
| Decision Tree | Chia không gian theo ngưỡng | Diễn giải trực quan | Variance rất cao | `max_depth`, `min_samples_leaf`, `ccp_alpha` |
| **Random Forest** | Bagging cây sâu + chọn ngẫu nhiên feature | Bền vững, ít cần tuning | Kém calibrate (dồn về giữa), mô hình lớn | `n_estimators`, `max_features`, `min_samples_leaf` |
| **Gradient Boosting** (HGB, LightGBM, XGBoost, CatBoost) | Cộng dồn cây nông sửa sai số theo gradient | **Mạnh nhất cho dữ liệu bảng** | Nhiều tham số, dễ overfit nếu không early stopping | `learning_rate`, `num_leaves`/`max_depth`, `min_child_samples`, subsampling, L1/L2 |
| SVM | Siêu phẳng lề cực đại (+ kernel) | Tốt với dữ liệu vừa, chiều cao | O(n²–n³), không ra xác suất trực tiếp | `C`, `gamma`, `kernel` |
| KNN | Bỏ phiếu láng giềng | Không giả định | Chậm khi dự đoán, nhạy thang đo | `n_neighbors`, `weights`, metric |
| Naive Bayes | Bayes + giả định độc lập có điều kiện | Rất nhanh, văn bản | Giả định mạnh, xác suất cực đoan | `alpha` (smoothing) |
| Neural Network (MLP) | Hàm hợp phi tuyến | Đa phương thức, dữ liệu lớn | Cần nhiều dữ liệu và tuning | Kiến trúc, `learning_rate`, `alpha`, early stopping |

### Logistic Regression trong scikit-learn ≥ 1.8: đặc tả penalty mới

**[Docs]** Từ phiên bản 1.8, tham số `penalty` của `LogisticRegression` **bị deprecate** (sẽ xóa ở 1.10). Thay bằng `l1_ratio` + `C`: `l1_ratio=0` là L2 (mặc định), `l1_ratio=1` là L1, `0 < l1_ratio < 1` là Elastic-Net, `C=np.inf` là không regularization. Bảng tương thích solver:

| solver | l1_ratio hỗ trợ | Đa lớp multinomial | Ghi chú tài liệu |
|---|---|---|---|
| `lbfgs` (mặc định) | 0 | ✅ | Mặc định tốt cho đa số bài toán |
| `liblinear` | 0 hoặc 1 | ❌ | Tốt cho dữ liệu nhỏ |
| `newton-cg` | 0 | ✅ | |
| `newton-cholesky` | 0 | ✅ | Tốt khi n_samples ≫ n_features, nhiều one-hot hiếm |
| `sag` | 0 | ✅ | Nhanh với dữ liệu lớn; cần feature cùng thang |
| `saga` | 0 → 1 | ✅ | Duy nhất hỗ trợ Elastic-Net |

```python
logit_variants = {
    "L2 (lbfgs)": LogisticRegression(C=1.0, max_iter=3000),
    "L1 (liblinear)": LogisticRegression(C=0.1, l1_ratio=1, solver="liblinear", max_iter=3000),
    "ElasticNet (saga)": LogisticRegression(C=0.1, l1_ratio=0.5, solver="saga", max_iter=5000),
    "Không penalty": LogisticRegression(C=np.inf, max_iter=3000),
}
for name, est in logit_variants.items():
    pipe = make_pipeline(build_preprocessor(scale=True), est).fit(X_train, y_train)
    coef = pipe[-1].coef_.ravel()
    print(f"{name:18s} PR-AUC={average_precision_score(y_valid, pipe.predict_proba(X_valid)[:, 1]):.4f}"
          f"  hệ số = 0: {(np.abs(coef) < 1e-8).sum()}/{coef.size}")
```

## 8.4. So sánh nhiều mô hình trên cùng một CV

```python
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

candidates = {
    "logreg": make_pipeline(build_preprocessor(scale=True), LogisticRegression(max_iter=3000)),
    "random_forest": make_pipeline(build_preprocessor(scale=False), RandomForestClassifier(
        n_estimators=300, min_samples_leaf=10, max_features="sqrt", n_jobs=1, random_state=0)),
    "hist_gb": make_pipeline(build_preprocessor(scale=False), HistGradientBoostingClassifier(
        learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, max_iter=300, random_state=0)),
    "lightgbm": make_pipeline(build_preprocessor(scale=False), lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1, verbose=-1, random_state=0)),
    "xgboost": make_pipeline(build_preprocessor(scale=False), XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=5, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", random_state=0, n_jobs=1)),
    "catboost": make_pipeline(build_preprocessor(scale=False), CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=5, random_seed=0, verbose=0, thread_count=1)),
}
rows = []
for name, model in candidates.items():
    # Song song ở cấp fold (joblib), mỗi mô hình 1 luồng: tránh oversubscription (mục 8.11)
    r = cross_validate(model, X_train, y_train, cv=cv, n_jobs=-1,
                       scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "log_loss": "neg_log_loss"})
    rows.append({"model": name, "roc_auc": r["test_roc_auc"].mean(), "pr_auc": r["test_pr_auc"].mean(),
                 "pr_auc_std": r["test_pr_auc"].std(), "log_loss": -r["test_log_loss"].mean(),
                 "fit_s": r["fit_time"].mean()})
leaderboard = pd.DataFrame(rows).set_index("model").sort_values("pr_auc", ascending=False)
print(leaderboard.round(4))
```

> **Đọc kết quả với tư duy thống kê.** Cơ chế sinh dữ liệu mô phỏng **gần tuyến tính trên thang log-odds** (Chương 0.8), nên Logistic Regression đứng ngang hoặc trên các mô hình boosting. Đây là bài học quan trọng: GBM không tự động thắng. Chênh lệch giữa các mô hình nhỏ hơn độ lệch chuẩn giữa các fold thì **chưa đủ bằng chứng** để chọn mô hình phức tạp hơn (Chương 9.10). Trong dữ liệu thật có tương tác và phi tuyến mạnh, GBM thường dẫn đầu.

## 8.5. Gradient Boosting chuyên sâu

### Thuật toán (Friedman, 2001) và cải tiến bậc hai (XGBoost)

Khởi tạo $F_0(x) = \arg\min_c \sum L(y_i, c)$. Tại bước *m*:

1. Tính **pseudo-residual** $g_i = \partial L(y_i, F)/\partial F\,|_{F_{m-1}}$ (và Hessian $h_i$ với XGBoost/LightGBM).
2. Khớp cây $h_m$ để dự đoán $-g_i$. Giá trị lá tối ưu theo xấp xỉ Taylor bậc hai là $w_j^* = -\dfrac{\sum_{i\in j} g_i}{\sum_{i\in j} h_i + \lambda}$.
3. Cập nhật $F_m = F_{m-1} + \eta\, h_m$, với $\eta$ là **learning rate (shrinkage)**.

Gain khi tách một nút (XGBoost): $\frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$. Đây là nguồn gốc ý nghĩa của các tham số `reg_lambda` (λ), `min_split_gain`/`gamma` (γ), `min_child_weight` (tổng Hessian tối thiểu ở lá).

### So sánh các triển khai

| | HistGradientBoosting (sklearn) | LightGBM | XGBoost | CatBoost |
|---|---|---|---|---|
| Tăng trưởng cây | Leaf-wise (`max_leaf_nodes`) | **Leaf-wise** | Depth-wise (mặc định), `grow_policy="lossguide"` | **Symmetric (oblivious)** |
| Histogram binning | ✅ (255 bins) | ✅ | ✅ (`tree_method="hist"`) | ✅ |
| Biến phân loại gốc | `categorical_features="from_dtype"` | ✅ | `enable_categorical=True` | ✅ **Ordered target statistics** |
| Missing gốc | ✅ | ✅ | ✅ | ✅ |
| Monotonic constraints | `monotonic_cst` | `monotone_constraints` | `monotone_constraints` | `monotone_constraints` |
| Interaction constraints | `interaction_cst` | `interaction_constraints` | `interaction_constraints` | — |
| Early stopping mặc định | **Bật nếu n > 10 000** | Cần validation set | Cần validation set | `use_best_model` |
| Điểm đặc biệt | Không phụ thuộc ngoài sklearn | Nhanh nhất, GOSS, EFB | Hệ sinh thái lớn, GPU | Ít overfit với biến phân loại, ít cần tuning |

### Hướng dẫn tuning chính thức của LightGBM (*Parameters Tuning*)

**[Docs]** LightGBM dùng **leaf-wise growth**: hội tụ nhanh hơn depth-wise nhưng dễ overfit nếu tham số không phù hợp. Ba tham số quan trọng nhất:

1. **`num_leaves`**: tham số chính điều khiển độ phức tạp. Về lý thuyết `num_leaves = 2^max_depth` tương đương cây depth-wise, nhưng thực tế **nên nhỏ hơn**. Ví dụ tài liệu đưa ra: `max_depth=7` thì 127 lá có thể overfit, còn 70–80 lá có thể tốt hơn.
2. **`min_data_in_leaf`** (`min_child_samples`): rất quan trọng để chống overfit. Với dữ liệu lớn, đặt hàng trăm đến hàng nghìn.
3. **`max_depth`**: giới hạn độ sâu tường minh. Nếu đặt thì nên đặt `num_leaves ≤ 2^max_depth`.

| Mục tiêu | Khuyến nghị của tài liệu LightGBM |
|---|---|
| **Chính xác hơn** | `max_bin` lớn; `learning_rate` nhỏ + `num_iterations` lớn; `num_leaves` lớn (rủi ro overfit); thêm dữ liệu; thử `dart` |
| **Chống overfit** | `max_bin` nhỏ; `num_leaves` nhỏ; `min_data_in_leaf` và `min_sum_hessian_in_leaf`; bagging (`bagging_fraction` + `bagging_freq`); `feature_fraction`; `lambda_l1`, `lambda_l2`, `min_gain_to_split`; `max_depth`; `extra_trees`; tăng `path_smooth` |
| **Nhanh hơn** | Ít lá/ít tầng; tăng `min_gain_to_split`, `min_data_in_leaf`; early stopping; giảm `max_bin`; giảm `feature_fraction`; bagging; số luồng = số **lõi vật lý** |

### Hướng dẫn tuning chính thức của XGBoost (*Notes on Parameter Tuning*)

**[Docs]** Có hai hướng kiểm soát overfit: (1) **trực tiếp giới hạn độ phức tạp**: `max_depth`, `min_child_weight`, `gamma`; (2) **thêm ngẫu nhiên**: `subsample`, `colsample_bytree`, giảm `eta` (và tăng số vòng). Với **dữ liệu mất cân bằng**:

- Nếu chỉ quan tâm **thứ hạng (AUC)**: cân bằng trọng số bằng `scale_pos_weight` và đánh giá bằng AUC.
- Nếu cần **xác suất đúng**: **không** cân bằng lại dữ liệu. Đặt `max_delta_step` hữu hạn (ví dụ 1) để giúp hội tụ.

### Early stopping đúng cách

```python
prep = build_preprocessor(scale=False)
Xtr = prep.fit_transform(X_train, y_train)          # fit tiền xử lý CHỈ trên train
Xva = prep.transform(X_valid)
feature_names = list(prep.get_feature_names_out())

booster = lgb.LGBMClassifier(
    n_estimators=5_000, learning_rate=0.02, num_leaves=15, min_child_samples=80,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    verbose=-1, random_state=0,
)
booster.fit(
    pd.DataFrame(Xtr, columns=feature_names), y_train,
    eval_set=[(pd.DataFrame(Xva, columns=feature_names), y_valid)],
    eval_metric=["average_precision", "binary_logloss"],
    callbacks=[lgb.early_stopping(stopping_rounds=200, first_metric_only=True, verbose=False),
               lgb.log_evaluation(period=0)],
)
hist = booster.evals_result_["valid_0"]
print("best_iteration_ =", booster.best_iteration_,
      "| PR-AUC tốt nhất =", round(max(hist["average_precision"]), 4))
```

> **Lưu ý.** Tập dùng cho early stopping đã "bị nhìn", nên **không** dùng nó để báo cáo điểm cuối. Khi huấn luyện lại trên train + valid, đặt `n_estimators ≈ best_iteration × (1 + tỷ lệ dữ liệu tăng thêm)`.

### XGBoost và CatBoost với early stopping

```python
xgb_model = XGBClassifier(
    n_estimators=5_000, learning_rate=0.02, max_depth=4, min_child_weight=5, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", eval_metric="aucpr",
    early_stopping_rounds=200, random_state=0, n_jobs=2,
)
xgb_model.fit(Xtr, y_train, eval_set=[(Xva, y_valid)], verbose=False)
print("XGBoost best_iteration =", xgb_model.best_iteration,
      "| PR-AUC valid =", round(average_precision_score(y_valid, xgb_model.predict_proba(Xva)[:, 1]), 4))

# CatBoost: đưa thẳng cột phân loại dạng chuỗi; ordered boosting giảm target leakage nội bộ
cat_cols = ["contract", "payment_method", "region"]
num_raw = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls", "data_usage_gb"]


def to_catboost(X: pd.DataFrame) -> pd.DataFrame:
    out = X[num_raw + cat_cols].copy()
    out[cat_cols] = out[cat_cols].fillna("missing").astype(str)
    return out


cb = CatBoostClassifier(iterations=3_000, learning_rate=0.03, depth=5, l2_leaf_reg=3, eval_metric="PRAUC",
                        early_stopping_rounds=200, random_seed=0, verbose=0, thread_count=2)
cb.fit(to_catboost(X_train), y_train, cat_features=cat_cols, eval_set=(to_catboost(X_valid), y_valid),
       use_best_model=True)
print("CatBoost best_iteration =", cb.get_best_iteration(),
      "| PR-AUC valid =", round(average_precision_score(y_valid, cb.predict_proba(to_catboost(X_valid))[:, 1]), 4))
```

### Monotonic constraints: mô hình phải "hợp lý" về nghiệp vụ

**[Docs]** `monotonic_cst` của `HistGradientBoosting*` nhận giá trị 1 (đồng biến), −1 (nghịch biến), 0 (không ràng buộc) cho từng feature. Ràng buộc giúp mô hình **hợp lý về nghiệp vụ**, **bền vững với nhiễu** và **dễ được duyệt** (ngân hàng, bảo hiểm).

```python
mono_features = ["support_calls", "tenure_months", "monthly_charges", "age"]
Xm_tr, Xm_va = X_train[mono_features], X_valid[mono_features]
free = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xm_tr, y_train)
mono = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0,
                                      monotonic_cst={"support_calls": 1, "tenure_months": -1}).fit(Xm_tr, y_train)

grid_calls = Xm_va.median().to_frame().T.loc[np.repeat(0, 9)].assign(support_calls=np.arange(9))
for name, m in [("không ràng buộc", free), ("monotonic", mono)]:
    p = m.predict_proba(grid_calls)[:, 1]
    print(f"{name:16s} P(churn) theo support_calls 0..8: {np.round(p, 3)} | đơn điệu: {bool(np.all(np.diff(p) >= -1e-12))}"
          f" | PR-AUC={average_precision_score(y_valid, m.predict_proba(Xm_va)[:, 1]):.4f}")
# Dữ liệu mô phỏng ít nhiễu nên mô hình tự do cũng đã gần đơn điệu. Trên dữ liệu thật, ràng buộc loại bỏ
# các đoạn "răng cưa" phi lý (thêm 1 cuộc gọi khiếu nại mà rủi ro lại giảm) và thường còn tăng tính tổng quát.
```

## 8.6. Tối ưu siêu tham số

| Phương pháp | Ý tưởng | Ưu | Nhược | Công cụ |
|---|---|---|---|---|
| Grid search | Thử mọi tổ hợp | Toàn diện, song song tốt | Bùng nổ tổ hợp | `GridSearchCV` |
| **Random search** | Lấy mẫu ngẫu nhiên | Hiệu quả hơn grid khi ít tham số thực sự quan trọng (Bergstra & Bengio, 2012) | Không học từ lịch sử | `RandomizedSearchCV` |
| Successive halving / Hyperband | Cấp ít tài nguyên cho nhiều cấu hình, loại dần | Tiết kiệm lớn | Cấu hình "khởi động chậm" bị loại oan | `HalvingRandomSearchCV`, `HyperbandPruner` |
| **Bayesian (TPE, GP)** | Mô hình hóa quan hệ tham số → điểm | Ít lần thử | Tuần tự hơn | **Optuna**, Hyperopt |
| CMA-ES | Chiến lược tiến hóa | Tốt với tham số liên tục, nhiều trial | | `CmaEsSampler` |

**Nguyên tắc thực hành:**

1. Không tune `n_estimators`: dùng **early stopping**.
2. Không gian tìm kiếm theo **thang log** cho learning rate và regularization.
3. Cố định **ngân sách** (số trial, thời gian). Tuning thường chỉ thêm 1–3%.
4. Tuning dùng **CV hoặc validation**; báo cáo trên dữ liệu chưa thấy (Chương 7.8).

### Random search và Successive Halving (scikit-learn)

```python
from scipy.stats import loguniform, randint, uniform
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV

lgbm_pipe = Pipeline([("prep", build_preprocessor(scale=False)),
                      ("clf", lgb.LGBMClassifier(n_estimators=300, subsample_freq=1, n_jobs=1, verbose=-1,
                                                 random_state=0))])
space = {
    "clf__num_leaves": randint(7, 64),
    "clf__min_child_samples": randint(20, 300),
    "clf__learning_rate": loguniform(0.01, 0.2),
    "clf__colsample_bytree": uniform(0.5, 0.5),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__reg_lambda": loguniform(1e-3, 10),
}
t0 = time.perf_counter()
rs = RandomizedSearchCV(lgbm_pipe, space, n_iter=12, cv=StratifiedKFold(3, shuffle=True, random_state=0),
                        scoring="average_precision", random_state=0, n_jobs=-1).fit(X_train, y_train)
print(f"Random search: best={rs.best_score_:.4f} ({time.perf_counter() - t0:.0f}s)")

t0 = time.perf_counter()
hs = HalvingRandomSearchCV(lgbm_pipe, space, n_candidates=24, factor=3, resource="clf__n_estimators",
                           min_resources=30, max_resources=300, cv=StratifiedKFold(3, shuffle=True, random_state=0),
                           scoring="average_precision", random_state=0, n_jobs=-1).fit(X_train, y_train)
print(f"Halving search: best={hs.best_score_:.4f} | {hs.n_candidates_} ứng viên qua các vòng "
      f"({time.perf_counter() - t0:.0f}s)")
```

### Optuna: Bayesian optimization với pruning

**[Docs]** Các khái niệm của Optuna: **Study** (một phiên tối ưu), **Trial** (một lần thử), **Sampler** (đề xuất tham số: `TPESampler` mặc định, `GPSampler`, `CmaEsSampler`, `NSGAIISampler` cho đa mục tiêu), **Pruner** (dừng sớm trial kém: `MedianPruner`, `SuccessiveHalvingPruner`, `HyperbandPruner`, `WilcoxonPruner`), **Storage** (RDB để lưu, tiếp tục và chạy phân tán).

```python
import optuna
from optuna.importance import get_param_importances

optuna.logging.set_verbosity(optuna.logging.WARNING)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
fold_data = []
for tr, va in inner_cv.split(X_train, y_train):                  # tiền xử lý mỗi fold một lần, dùng lại
    p = build_preprocessor(scale=False)
    fold_data.append((p.fit_transform(X_train.iloc[tr], y_train.iloc[tr]), y_train.iloc[tr],
                      p.transform(X_train.iloc[va]), y_train.iloc[va]))


def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": 2_000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 400, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "verbose": -1, "random_state": 0,
    }
    scores = []
    for step, (xtr, ytr, xva, yva) in enumerate(fold_data):
        m = lgb.LGBMClassifier(**params).fit(
            xtr, ytr, eval_set=[(xva, yva)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False)])
        scores.append(average_precision_score(yva, m.predict_proba(xva)[:, 1]))
        trial.set_user_attr(f"best_iter_{step}", int(m.best_iteration_))
        trial.report(float(np.mean(scores)), step=step)            # báo cáo kết quả trung gian
        if trial.should_prune():                                    # pruner quyết định dừng sớm
            raise optuna.TrialPruned()
    return float(np.mean(scores))


study = optuna.create_study(
    direction="maximize", study_name="churn-lgbm",
    sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, n_startup_trials=8),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    storage="sqlite:///optuna.db", load_if_exists=True,            # lưu lịch sử, tiếp tục được, chạy song song được
)
study.optimize(objective, n_trials=25, timeout=600)
pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
print(f"Best PR-AUC={study.best_value:.4f} | trials={len(study.trials)} (pruned {pruned})")
print("Best params:", {k: round(v, 4) if isinstance(v, float) else v for k, v in study.best_params.items()})
print("Tầm quan trọng tham số (fANOVA):", {k: round(v, 3) for k, v in get_param_importances(study).items()})
```

```python norun
# Trực quan hóa (cần plotly) và dashboard
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
optuna.visualization.plot_parallel_coordinate(study).show()
# optuna-dashboard sqlite:///optuna.db   -> giao diện web theo dõi study
```

**Đa mục tiêu** (ví dụ tối đa PR-AUC và tối thiểu độ trễ suy luận): `optuna.create_study(directions=["maximize", "minimize"])` với `NSGAIISampler`, rồi chọn từ **mặt Pareto** `study.best_trials`.

## 8.7. Ensemble

| Kỹ thuật | Cơ chế | Hiệu quả khi |
|---|---|---|
| Bagging | Trung bình mô hình trên mẫu bootstrap | Giảm variance (cây sâu) |
| Boosting | Mô hình sau sửa lỗi mô hình trước | Giảm bias |
| **Soft voting / blending** | Trung bình (có trọng số) xác suất | Các mô hình **đa dạng**, lỗi ít tương quan |
| Rank averaging | Trung bình thứ hạng | Metric dựa trên thứ hạng (AUC) |
| **Stacking** | Meta-model học trên **OOF predictions** | Thi đấu; production cần cân nhắc chi phí |
| Seed averaging | Cùng mô hình, nhiều seed | Ổn định dự đoán, rẻ |

**[Docs]** `StackingClassifier` dùng `cross_val_predict` nội bộ (tham số `cv`) để tạo đặc trưng cho `final_estimator`, nhờ vậy meta-model không học trên dự đoán bị overfit. Sau đó các base estimator được fit lại trên toàn bộ dữ liệu.

```python
from sklearn.ensemble import StackingClassifier, VotingClassifier

base = [("logreg", candidates["logreg"]), ("lightgbm", candidates["lightgbm"]), ("catboost", candidates["catboost"])]
voting = VotingClassifier(base, voting="soft", weights=[2, 1, 1])
stacking = StackingClassifier(base, final_estimator=LogisticRegression(C=1.0), cv=StratifiedKFold(5, shuffle=True, random_state=1),
                              stack_method="predict_proba")
for name, m in [("voting", voting), ("stacking", stacking)]:
    p = m.fit(X_train, y_train).predict_proba(X_valid)[:, 1]
    print(f"{name:9s} PR-AUC valid={average_precision_score(y_valid, p):.4f} | ROC-AUC={roc_auc_score(y_valid, p):.4f}")

# Tương quan dự đoán giữa các base model: thấp -> ensemble có lợi
preds = pd.DataFrame({n: m.fit(X_train, y_train).predict_proba(X_valid)[:, 1] for n, m in base})
print(preds.corr(method="spearman").round(3))
```

> **Production trade-off.** Stacking 3 mô hình nghĩa là 3 lần độ trễ, 3 lần bảo trì và khó giải thích hơn. Chỉ dùng khi lợi ích kinh doanh đo được vượt chi phí.

## 8.8. Chẩn đoán: learning curve và validation curve

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay, ValidationCurveDisplay

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
LearningCurveDisplay.from_estimator(
    candidates["lightgbm"], X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 6),
    cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="average_precision",
    score_type="both", n_jobs=-1, ax=axes[0])
axes[0].set_title("Learning curve: còn tăng khi thêm dữ liệu?")
ValidationCurveDisplay.from_estimator(
    candidates["lightgbm"], X_train, y_train, param_name="lgbmclassifier__num_leaves",
    param_range=[3, 7, 15, 31, 63, 127], cv=StratifiedKFold(3, shuffle=True, random_state=0),
    scoring="average_precision", n_jobs=-1, ax=axes[1])
axes[1].set_xscale("log", base=2)
axes[1].set_title("Validation curve: độ phức tạp tối ưu")
plt.tight_layout()
```

| Hình dạng learning curve | Chẩn đoán | Hành động |
|---|---|---|
| Train cao, validation thấp, khoảng cách lớn, validation còn tăng | High variance | **Thêm dữ liệu**, regularization |
| Hai đường hội tụ ở mức thấp | High bias | Feature tốt hơn, mô hình mạnh hơn |
| Hai đường hội tụ ở mức chấp nhận được | Ổn | Thêm dữ liệu ít lợi ích; tập trung vào feature/nhãn |

### Phân tích lỗi (error analysis)

**[Course]** Andrew Ng: lấy khoảng 100 mẫu bị dự đoán sai trên dev set, phân loại thủ công theo nguyên nhân, đếm tỷ lệ, rồi ưu tiên sửa nhóm lỗi lớn nhất. Với dữ liệu bảng, dùng phiên bản tự động: **slice analysis** (Chương 9.11) để tìm phân khúc có lỗi cao bất thường.

## 8.9. Neural network cho dữ liệu bảng

**Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022)** chỉ ra mô hình cây vẫn vượt deep learning trên dữ liệu bảng cỡ vừa, vì: (1) NN thiên về hàm trơn còn dữ liệu bảng thường có hàm "bậc thang"; (2) NN nhạy với feature vô dụng; (3) NN không bất biến với phép quay. Deep learning đáng thử khi dữ liệu rất lớn, đa phương thức (bảng + văn bản + ảnh), hoặc cần embedding cho biến cardinality cực cao. Các kiến trúc chuyên dụng: **FT-Transformer**, **TabNet**, **TabPFN** (rất mạnh với dữ liệu nhỏ).

```python
from sklearn.neural_network import MLPClassifier

mlp = make_pipeline(build_preprocessor(scale=True), MLPClassifier(
    hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, batch_size=256,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=10, max_iter=300, random_state=0))
mlp.fit(X_train, y_train)
print(f"MLP: {mlp[-1].n_iter_} epochs | PR-AUC valid={average_precision_score(y_valid, mlp.predict_proba(X_valid)[:, 1]):.4f}")
```

Vòng huấn luyện PyTorch chuẩn mực (khối minh họa, không chạy trong môi trường kiểm chứng vì không cài PyTorch):

```python norun
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TabularMLP(nn.Module):
    def __init__(self, n_in: int, hidden=(128, 64), dropout: float = 0.2):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)                    # logits


def train(model, train_dl, X_val, y_val, epochs=100, patience=10, lr=1e-3, device="cpu"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = nn.BCEWithLogitsLoss()                      # ổn định số hơn sigmoid + BCELoss
    best, best_state, wait = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.inference_mode():
            p = torch.sigmoid(model(X_val.to(device))).cpu().numpy()
        score = average_precision_score(y_val, p)
        sched.step(score)
        if score > best + 1e-4:
            best, wait = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif (wait := wait + 1) >= patience:
            break
    model.load_state_dict(best_state)
    return model, best
```

## 8.10. Huấn luyện tái lập được, có tracking (MLflow)

Package tham chiếu có lệnh huấn luyện production `python -m churn.models.train --config configs/train.yaml`. Lệnh này thực hiện tuần tự: đọc config (pydantic), nạp dữ liệu, làm sạch, kiểm tra **data contract**, **chia theo thời gian + kiểm tra leakage**, huấn luyện pipeline, tính metric, rồi lưu `model.joblib` + `metrics.json`. Toàn bộ params, metrics, git commit và mô hình được **log vào MLflow**. Chạy trực tiếp từ Python:

```python
import json
import os
from pathlib import Path

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"      # MLflow 3: file store ./mlruns ở chế độ bảo trì
Path("configs").mkdir(exist_ok=True)
Path("configs/train.yaml").write_text("""
target: churn
random_state: 42
split: {date_col: signup_date, train_end: "2023-10-01", valid_end: "2024-01-01", gap_days: 0}
model_params: {n_estimators: 400, learning_rate: 0.03, num_leaves: 15, min_child_samples: 50,
               subsample: 0.8, subsample_freq: 1, colsample_bytree: 0.8, reg_lambda: 1.0}
""")

from churn.models.train import main as train_main

metrics = train_main("configs/train.yaml", out_dir=".")
print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=1))

import mlflow

runs = mlflow.search_runs(experiment_names=["churn-prediction"])
print(runs[["run_id", "metrics.valid_pr_auc", "params.num_leaves", "tags.git_commit"]].head(3))
```

**[Docs]** Những điểm của MLflow 3 cần biết:

- `mlflow.sklearn.log_model(sk_model=..., name="model")`: tham số `artifact_path` được thay bằng `name`. Mô hình trở thành một **LoggedModel** có `model_id`, nạp bằng `models:/<model_id>`.
- Backend file `./mlruns` ở **chế độ bảo trì**: dùng `sqlite:///mlflow.db` hoặc PostgreSQL.
- Ngoài Databricks, scikit-learn được serialize mặc định bằng **skops** (an toàn khi nạp). Pipeline có custom class hoặc LightGBM booster phải khai báo `skops_trusted_types` hoặc dùng `serialization_format="cloudpickle"` kèm `code_paths`. Khi đó chỉ nạp model từ registry đáng tin.

### Chiến lược mô hình cuối cùng

1. Chọn cấu hình bằng CV/validation (đã tune, có early stopping).
2. **Đánh giá một lần** trên test out-of-time, báo cáo có CI (Chương 9).
3. **Huấn luyện lại** trên toàn bộ dữ liệu có nhãn mới nhất (train + valid + test) với cấu hình đã cố định. Số cây hiệu chỉnh theo lượng dữ liệu.
4. Đăng ký vào registry và triển khai qua shadow/canary (Chương 12).

## 8.11. Song song hóa và số luồng CPU: bài học từ chính môi trường viết handbook

Khi chạy kiểm chứng chương này trong container (báo `nproc` = 4), `HistGradientBoostingClassifier` với 300 vòng trên 14 000 dòng mất **17–146 giây**. Giới hạn OpenMP còn 1–2 luồng thì chỉ mất **0.2 giây**. Nguyên nhân: container được cấp **hạn mức CPU (cgroup quota) thấp hơn số vCPU nhìn thấy**. Các luồng OpenMP *spin-wait* chờ nhau trong khi bị hệ điều hành tạm dừng, nên hiệu năng sụp đổ. Hiện tượng này rất phổ biến trên Kubernetes, CI runner và notebook dùng chung.

| Nguồn song song | Điều khiển bởi | Mặc định |
|---|---|---|
| joblib (CV, grid search, RF) | `n_jobs` | 1 (None); `-1` = mọi CPU |
| OpenMP (HistGB, LightGBM, XGBoost) | `OMP_NUM_THREADS`, `n_jobs`/`num_threads`/`nthread` | Mọi CPU |
| BLAS (numpy, scipy, Logistic lbfgs) | `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | Mọi CPU |
| CatBoost | `thread_count` | Mọi CPU |

**[Docs]** scikit-learn *Parallelism, resource management, and configuration*: tổng số luồng = (số tiến trình joblib) × (số luồng OpenMP/BLAS mỗi tiến trình). Khi vượt số lõi thật sẽ xảy ra **oversubscription**. scikit-learn tự giới hạn luồng của tiến trình con loky, nhưng không kiểm soát được thư viện ngoài. **[Docs]** LightGBM: đặt `num_threads` bằng số **lõi vật lý**, không phải số luồng hyper-threading.

Phép đo thứ hai trong cùng môi trường: `cross_val_score(..., n_jobs=2)` với pipeline LightGBM để số luồng mặc định mất **451 giây**, trong khi một lần `fit` đơn lẻ chỉ mất **0.47 giây**. Mỗi worker của joblib lại mở nhiều luồng OpenMP, nên tổng số luồng vượt xa số lõi thật. Đặt `LGBMClassifier(n_jobs=1)` khi đã song song ở cấp fold sẽ khắc phục hoàn toàn.

Quy tắc thực hành:

1. **Song song ở một cấp:** hoặc nhiều fold (`n_jobs=-1` ở `cross_validate`/`GridSearchCV`) với mô hình 1 luồng, hoặc 1 fold với mô hình đa luồng. Không làm cả hai.
2. Trong container, đặt `OMP_NUM_THREADS` (và `n_jobs` của mô hình) bằng **số CPU được cấp**. Trên Kubernetes, giá trị này là `resources.limits.cpu`.
3. Kiểm tra bằng `threadpoolctl.threadpool_info()` và giới hạn cục bộ bằng `threadpool_limits(limits=1)`.

```python
from threadpoolctl import threadpool_info, threadpool_limits

print({(i["internal_api"], i["num_threads"]) for i in threadpool_info()})
with threadpool_limits(limits=1):                    # giới hạn OpenMP/BLAS trong một khối code
    t0 = time.perf_counter()
    HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(Xtr, y_train)
    print(f"HGB 1 luồng: {time.perf_counter() - t0:.2f}s")
```

> **Checklist Chương 8**
> - [ ] Có baseline ngây thơ và baseline nghiệp vụ; mô hình phải vượt rõ rệt.
> - [ ] Loss/metric nhất quán với đại lượng cần dự đoán (xác suất, mean, median, quantile).
> - [ ] Mọi mô hình được so sánh trên cùng CV, cùng metric, báo cáo mean ± std.
> - [ ] Early stopping dùng validation riêng; không tune `n_estimators` trực tiếp.
> - [ ] Tuning có ngân sách, không gian log-scale, lưu lịch sử (Optuna storage), seed cố định.
> - [ ] Đã xem learning/validation curve để quyết định thêm dữ liệu hay đổi mô hình.
> - [ ] Ràng buộc đơn điệu cho feature có quan hệ nghiệp vụ rõ ràng (khi cần).
> - [ ] Huấn luyện là một lệnh tái lập được; params, metrics, commit, mô hình được log vào MLflow.
> - [ ] Dùng API hiện hành (`l1_ratio` thay `penalty`; MLflow `name=`; TargetEncoder `cv=`).
> - [ ] Song song hóa ở một cấp; số luồng khớp CPU được cấp (tránh oversubscription trong container).

### Tài liệu tham khảo Chương 8

- scikit-learn User Guide: *Parallelism, resource management, and configuration*; *Linear Models* (Logistic regression, solvers); *Ensembles: Gradient boosting, random forests, bagging, voting, stacking* (HistGradientBoosting: early stopping, categorical, monotonic & interaction constraints); *Tuning the hyper-parameters* (Randomized, Successive Halving); *Validation curves*; *Which scoring function should I use?*
- LightGBM documentation: *Parameters*, *Parameters Tuning*, *Features* (leaf-wise, GOSS, EFB). lightgbm.readthedocs.io
- XGBoost documentation: *Notes on Parameter Tuning*, *Introduction to Boosted Trees*. xgboost.readthedocs.io
- CatBoost documentation: *Transforming categorical features*, *Ordered boosting*. catboost.ai/docs
- Optuna documentation: *Key Features* (samplers, pruners, storage, multi-objective). optuna.readthedocs.io
- MLflow 3 documentation: *Tracking*, *Models (LoggedModel)*, *scikit-learn flavor*. mlflow.org/docs
- Friedman, J. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics 29(5).
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD. · Ke, G. et al. (2017). *LightGBM.* NeurIPS. · Prokhorenkova, L. et al. (2018). *CatBoost: unbiased boosting with categorical features.* NeurIPS.
- Bergstra, J. & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization.* JMLR 13. · Akiba, T. et al. (2019). *Optuna.* KDD.
- Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS.
- Stanford CS229 Lecture Notes (supervised learning, regularization, bias–variance). · Ng, A. *Machine Learning Yearning*.
