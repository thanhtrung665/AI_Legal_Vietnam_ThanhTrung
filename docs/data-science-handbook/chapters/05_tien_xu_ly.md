# CHƯƠNG 5. TIỀN XỬ LÝ DỮ LIỆU & FEATURE ENGINEERING

> *"Coming up with features is difficult, time-consuming, requires expert knowledge. 'Applied machine learning' is basically feature engineering."* — Andrew Ng.

**Mục tiêu chương:** biến dữ liệu thô thành ma trận feature **sạch, giàu thông tin, không rò rỉ**, được đóng gói thành một `Pipeline` duy nhất dùng chung cho huấn luyện, đánh giá và production.

## 5.1. Nguyên tắc nền tảng: fit trên train, transform trên mọi tập

**[Docs]** scikit-learn *Common pitfalls*:

- *Inconsistent preprocessing.* Phép biến đổi dùng lúc train phải được áp dụng **y hệt** lúc test và trong production. Nếu không, không gian feature thay đổi và mô hình hoạt động sai.
- *Data leakage during pre-processing.* Ví dụ chính thức của scikit-learn: chọn feature bằng `SelectKBest` trên **toàn bộ** dữ liệu ngẫu nhiên (X và y độc lập) cho accuracy **0.76** thay vì ~0.5. Khi đặt cùng bước đó trong `Pipeline` và CV, accuracy về đúng ~0.5.
- Khuyến nghị: dùng **`Pipeline`** để mọi bước có trạng thái (stateful) chỉ được `fit` trên phần train của từng fold.

| Loại bước | Ví dụ | Làm trước khi chia dữ liệu? |
|---|---|---|
| **Stateless** (không học gì từ dữ liệu) | Chuẩn hóa chuỗi, parse ngày, sửa giá trị vô lý → NaN, dedup chính xác, feature tỷ lệ `a/b` | ✅ Được |
| **Stateful** (học tham số) | Impute mean/median, scaling, encoding, binning theo phân vị, chọn feature, PCA, SMOTE | ❌ Phải nằm trong `Pipeline` |

```text
Dữ liệu thô ─▶ clean_churn (stateless) ─▶ split train/valid/test ─▶ Pipeline[ ChurnFeatures ─▶ ColumnTransformer(
                                                                       num: Winsorizer → SimpleImputer(+indicator) → Scaler
                                                                       low-card: Imputer → OneHotEncoder
                                                                       high-card: Imputer → TargetEncoder (cross-fitting)
                                                                     ) ─▶ Model ]
```

### Thiết lập

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data.split import time_split
from churn.data.synthetic import make_churn_data
from churn.features.build import RAW_FEATURES, build_preprocessor
from churn.features.clean import clean_churn

pd.set_option("display.width", 160)
raw = make_churn_data(n=20_000, seed=42)
df = clean_churn(raw)
train_df, valid_df, test_df = time_split(df, "signup_date", "2023-10-01", "2024-01-01")
X_train, y_train = train_df[RAW_FEATURES], train_df["churn"]
X_valid, y_valid = valid_df[RAW_FEATURES], valid_df["churn"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print(len(X_train), len(X_valid), len(test_df))
```

## 5.2. Làm sạch dữ liệu (stateless)

Mã nguồn đầy đủ ở `code/src/churn/features/clean.py`. Các quy tắc chính:

| Vấn đề | Quy tắc | Lý do |
|---|---|---|
| Trùng chính xác | `drop_duplicates()` | Bản ghi lặp làm lệch thống kê và gây leakage giữa train/test |
| Trùng theo khóa nghiệp vụ | Giữ bản ghi **mới nhất** theo thời gian | Một khách hàng = một dòng (theo grain đã định) |
| Chuỗi không nhất quán | Unicode NFC → strip → gộp khoảng trắng → lowercase → bảng ánh xạ alias | `"Month-to-Month "` = `"month-to-month"` |
| Giá trị **không thể có** | Chuyển thành `NaN` (tuổi 150, cước ≤ 0) | Để imputer xử lý thống nhất; **không** xóa dòng âm thầm |
| Kiểu dữ liệu | `pd.to_datetime(errors="coerce")`, ép số | Lỗi parse trở thành NaN có thể đếm được |

```python
before = raw["contract"].value_counts().to_dict()
after = df["contract"].value_counts().to_dict()
print("Trước:", before)
print("Sau  :", after)
print("Số dòng:", len(raw), "->", len(df))
```

## 5.3. Xử lý giá trị thiếu (Imputation)

| Phương pháp | Khi nào dùng | scikit-learn |
|---|---|---|
| Xóa dòng | Thiếu rất ít, MCAR, dữ liệu dồi dào | `dropna` (chỉ trên train; production vẫn phải xử lý NaN) |
| Xóa cột | Thiếu > 60–70% **và** không mang tín hiệu | — |
| Mean / Median | Baseline; median bền với ngoại lai | `SimpleImputer(strategy="median")` |
| Hằng số / mode | Biến phân loại | `SimpleImputer(strategy="constant", fill_value="missing")` |
| **Cờ thiếu** | Thiếu mang thông tin (MAR/MNAR) | `SimpleImputer(add_indicator=True)`, `MissingIndicator` |
| KNN | Quan hệ cục bộ, dữ liệu vừa | `KNNImputer` (cần scale trước, O(n²)) |
| Iterative (MICE-like) | MAR, quan hệ đa biến | `IterativeImputer` (experimental) |
| Không impute | Mô hình xử lý NaN gốc | `HistGradientBoosting*`, `DecisionTree*`/`RandomForest*` (≥1.4), LightGBM, XGBoost, CatBoost |

**[Docs]** Về *Multiple vs. Single Imputation*: `IterativeImputer` lấy cảm hứng từ gói R **MICE** nhưng trả về **một** lần impute. Muốn multiple imputation thì chạy lặp với `sample_posterior=True` và các seed khác nhau. Tài liệu cũng ghi rõ: trong bối cảnh **dự đoán** (không cần định lượng bất định do thiếu), lợi ích của multiple imputation vẫn là câu hỏi mở. **[Docs]** Từ phiên bản 1.2, tham số `keep_empty_features` quyết định cột toàn NaN bị bỏ hay giữ (điền 0). Điều này ảnh hưởng tới số cột đầu ra.

### So sánh chiến lược impute bằng chính metric cuối cùng

```python
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (bật API experimental)
from sklearn.impute import IterativeImputer, KNNImputer

num_cols = ["age", "tenure_months", "monthly_charges", "support_calls", "data_usage_gb"]
Xn, yn = X_train[num_cols].iloc[:6000], y_train.iloc[:6000]          # mẫu con để chạy nhanh

strategies = {
    "median": SimpleImputer(strategy="median"),
    "median+indicator": SimpleImputer(strategy="median", add_indicator=True),
    "knn(k=10)": make_pipeline(StandardScaler(), KNNImputer(n_neighbors=10)),
    "iterative(ET)": IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=30, random_state=0),
                                      max_iter=5, random_state=0),
}
for name, imp in strategies.items():
    pipe = make_pipeline(imp, StandardScaler(), LogisticRegression(max_iter=2000))
    s = cross_val_score(pipe, Xn, yn, cv=cv, scoring="average_precision")
    print(f"{name:18s} PR-AUC = {s.mean():.4f} ± {s.std():.4f}")
s = cross_val_score(HistGradientBoostingClassifier(random_state=0), Xn, yn, cv=cv, scoring="average_precision")
print(f"{'HGB (NaN gốc)':18s} PR-AUC = {s.mean():.4f} ± {s.std():.4f}")
```

> **[Kinh nghiệm]** Chọn cách impute bằng **CV trên metric cuối cùng**, không phải bằng việc giá trị điền "giống thật" nhất. Trên dữ liệu MCAR này, bốn chiến lược impute cho kết quả gần như bằng nhau, nên chọn phương án **đơn giản nhất** (median + indicator). Với dữ liệu có cơ chế MAR mạnh, KNN/Iterative mới thể hiện ưu thế. `HistGradientBoosting` với tham số mặc định thấp hơn ở đây vì mô hình chưa được tinh chỉnh trên mẫu 6 000 dòng, quan hệ thật lại gần tuyến tính. Đừng kết luận "xử lý NaN gốc kém": so sánh này đổi cả *mô hình* chứ không chỉ cách impute.

## 5.4. Ngoại lai: clip có học từ train

Winsorizer trong package (`code/src/churn/features/transformers.py`) tuân thủ **hợp đồng estimator** của scikit-learn (*Developing scikit-learn estimators*):

```python norun
class Winsorizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, q_low: float = 0.01, q_high: float = 0.99):
        self.q_low = q_low            # __init__ CHỈ lưu tham số: không validate, không tính toán
        self.q_high = q_high

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        self.lower_ = np.nanquantile(arr, self.q_low, axis=0)    # thuộc tính học được: hậu tố "_"
        self.upper_ = np.nanquantile(arr, self.q_high, axis=0)
        return self                                               # fit trả về self

    def transform(self, X):
        check_is_fitted(self, ["lower_", "upper_"])
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)   # NaN được giữ nguyên
```

`OneToOneFeatureMixin` tự cung cấp `get_feature_names_out`, nhờ đó transformer hoạt động với `set_output(transform="pandas")`. Package có test `check_transformer_general` của scikit-learn để bảo đảm tuân thủ API.

```python
from churn.features.transformers import Winsorizer

w = Winsorizer(0.01, 0.99).set_output(transform="pandas").fit(X_train[["monthly_charges"]])
clipped = w.transform(X_valid[["monthly_charges"]])
print("Ngưỡng học từ train:", w.lower_.round(2), w.upper_.round(2))
print("Max trước/sau:", X_valid["monthly_charges"].max(), "->", clipped["monthly_charges"].max())
```

## 5.5. Mã hóa biến phân loại

| Encoder | Cardinality | Mô hình phù hợp | Ghi chú (theo tài liệu chính thức) |
|---|---|---|---|
| `OneHotEncoder` | Thấp | Tuyến tính, NN, SVM | `handle_unknown="infrequent_if_exist"`, `min_frequency`, `max_categories`, `drop="if_binary"`; `sparse_output` |
| `OrdinalEncoder` | Có thứ tự / mô hình cây | Cây, GBM | `categories=[[...]]` để cố định thứ tự; `handle_unknown="use_encoded_value"`, `encoded_missing_value` |
| **`TargetEncoder`** | **Cao** | Mọi mô hình | Có **cross-fitting** trong `fit_transform`; `smooth="auto"` (empirical Bayes) |
| Frequency/count | Cao | Cây | Không dùng target nên không có leakage từ target |
| Hashing | Rất cao (URL, ID) | Tuyến tính | `FeatureHasher`; chấp nhận va chạm hash |
| Native categorical | Bất kỳ | HistGB (`categorical_features="from_dtype"`), LightGBM, CatBoost | CatBoost dùng *ordered target statistics* |
| Embedding | Rất cao | Neural network | Học biểu diễn |

### OneHotEncoder với nhóm hiếm và giá trị chưa thấy

```python
ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                    sparse_output=False).set_output(transform="pandas")
ohe.fit(X_train[["contract"]].fillna("missing"))
print(ohe.categories_, "| infrequent:", ohe.infrequent_categories_)
print(ohe.transform(pd.DataFrame({"contract": ["one-year", "lifetime-plan"]})))
# Không có mức nào < 1% nên không tồn tại nhóm infrequent: mức lạ "lifetime-plan" được mã hóa toàn 0
# (hành vi giống handle_unknown="ignore"). Nếu có nhóm infrequent, mức lạ sẽ được gán vào nhóm đó.
```

### TargetEncoder: công thức và vì sao phải cross-fitting

**[Docs]** Với target nhị phân, mã hóa của mức *i*:

$$S_i = \lambda_i\frac{n_{iY}}{n_i} + (1-\lambda_i)\frac{n_Y}{n}, \qquad \lambda_i = \frac{n_i}{m + n_i}$$

Trong đó *m* là hệ số làm trơn. `smooth="auto"` ước lượng $m = \sigma_i^2/\tau^2$ theo empirical Bayes. **`fit(X, y).transform(X)` khác `fit_transform(X, y)`**: `fit_transform` chia train thành *k* fold, rồi mã hóa mỗi fold bằng thống kê học từ *k−1* fold còn lại. Cách này ngăn mô hình "học thuộc" nhãn qua các mức hiếm. Thí nghiệm dưới đây tái hiện ví dụ chính thức *Target Encoder's Internal Cross fitting* của scikit-learn: thêm một biến **hoàn toàn ngẫu nhiên** có 3 000 mức.

```python
from sklearn.preprocessing import TargetEncoder

from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
useful = ["tenure_months", "support_calls", "monthly_charges"]
sc = StandardScaler().fit(X_train[useful])
U_tr, U_va = sc.transform(X_train[useful]), sc.transform(X_valid[useful])
noise_tr = pd.DataFrame({"noise_id": rng.integers(0, 3000, len(X_train)).astype(str)})
noise_va = pd.DataFrame({"noise_id": rng.integers(0, 3000, len(X_valid)).astype(str)})

# SAI: fit trên train rồi transform chính train -> mức hiếm "nhớ" nhãn của chính dòng đó
te_wrong = TargetEncoder(target_type="binary").fit(noise_tr, y_train)
Z_wrong_tr = np.c_[U_tr, te_wrong.transform(noise_tr)]
# ĐÚNG: fit_transform -> mỗi fold được mã hóa bằng thống kê của các fold khác
te_right = TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=0))
Z_right_tr = np.c_[U_tr, te_right.fit_transform(noise_tr, y_train)]

for name, enc, Z_tr in [("fit + transform (leak)", te_wrong, Z_wrong_tr),
                        ("fit_transform (đúng)", te_right, Z_right_tr)]:
    m = LogisticRegression(max_iter=1000).fit(Z_tr, y_train)
    auc_tr = roc_auc_score(y_train, m.predict_proba(Z_tr)[:, 1])
    auc_va = roc_auc_score(y_valid, m.predict_proba(np.c_[U_va, enc.transform(noise_va)])[:, 1])
    print(f"{name:22s} AUC train={auc_tr:.3f} valid={auc_va:.3f} | hệ số của biến nhiễu={m.coef_[0, -1]:+.2f}")
# Bản leak: mô hình dồn trọng số vào biến nhiễu (AUC train ảo cao) -> AUC valid giảm rõ.
# Bản đúng: hệ số biến nhiễu ~0, AUC train ≈ valid.
```

> **Lưu ý phiên bản.** Từ scikit-learn **1.9**, `TargetEncoder(shuffle=..., random_state=...)` bị deprecate (sẽ xóa ở 1.11). Hãy truyền splitter vào `cv`, ví dụ `cv=StratifiedKFold(5, shuffle=True, random_state=0)`, như package tham chiếu.

## 5.6. Feature Engineering: nơi tạo ra khác biệt lớn nhất

### a) Feature nghiệp vụ (domain features)

`ChurnFeatures` trong package tạo các feature có ý nghĩa nghiệp vụ, tính **theo từng dòng** (stateless):

| Feature | Công thức | Giả thuyết nghiệp vụ |
|---|---|---|
| `avg_charge_per_month` | total / max(tenure, 1) | Giá trị trung bình thực trả |
| `charge_vs_expected` | total / (monthly × tenure) | Lệch so với kỳ vọng: thay đổi gói, khuyến mãi |
| `calls_per_year` | support_calls / (tenure/12) | Cường độ khiếu nại chuẩn hóa theo thời gian |
| `is_new_customer` | tenure ≤ 3 | Giai đoạn onboarding rủi ro cao |
| `signup_month`, `signup_dow` | Từ ngày đăng ký | Mùa vụ chiến dịch |

```python
from churn.features.transformers import ChurnFeatures

feat = ChurnFeatures().fit(X_train).transform(X_valid)
print(feat[["avg_charge_per_month", "charge_vs_expected", "calls_per_year", "is_new_customer"]]
      .describe().T.round(2))
```

> **Bẫy thật đã gặp khi viết handbook: "số ngày kể từ một ngày cố định".** Phiên bản đầu của package có feature `days_since_signup = reference_date − signup_date`, với `reference_date` học từ tập train (ngày lớn nhất = 2023-09-30). Trên tập validation (khách đăng ký *sau* ngày đó), feature này **âm** và nằm ngoài miền giá trị mô hình từng thấy:
>
> ```python
> ref = X_train["signup_date"].max()
> print((ref - X_valid["signup_date"]).dt.days.describe()[["min", "max"]])   # toàn giá trị âm
> ```
>
> Nguyên tắc: feature "thời gian trôi qua" phải đo **tới thời điểm chấm điểm của chính dòng đó** (snapshot date). Ở đây `tenure_months` đã làm đúng điều này, nên feature kia vừa sai vừa thừa và đã bị loại khỏi package. Adversarial validation (Chương 3.11) và kiểm tra miền giá trị feature giữa train/valid giúp phát hiện sớm lỗi này.

### b) Mã hóa chu kỳ (cyclical encoding)

Tháng 12 và tháng 1 gần nhau, nhưng mã hóa số nguyên đặt chúng xa nhau nhất. Ánh xạ lên vòng tròn đơn vị giải quyết điều này. **[Docs]** Ví dụ chính thức *Time-related feature engineering* của scikit-learn so sánh one-hot, sin/cos và `SplineTransformer(extrapolation="periodic")` cho dữ liệu theo giờ.

```python
from sklearn.preprocessing import FunctionTransformer, SplineTransformer


def sin_transformer(period: int) -> FunctionTransformer:
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi), feature_names_out="one-to-one")


def cos_transformer(period: int) -> FunctionTransformer:
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi), feature_names_out="one-to-one")


months = pd.DataFrame({"month": np.arange(1, 13)})
cyc = pd.DataFrame({"sin": sin_transformer(12).fit_transform(months)["month"],
                    "cos": cos_transformer(12).fit_transform(months)["month"]})
print("Khoảng cách 12↔1:", np.linalg.norm(cyc.iloc[11] - cyc.iloc[0]).round(3),
      "| 6↔7:", np.linalg.norm(cyc.iloc[5] - cyc.iloc[6]).round(3))    # bằng nhau: đúng tính chu kỳ
periodic_spline = SplineTransformer(degree=3, n_knots=13, knots=np.linspace(1, 13, 13).reshape(-1, 1),
                                    extrapolation="periodic")
print("Periodic spline features:", periodic_spline.fit_transform(months).shape)
```

### c) Tổng hợp theo cửa sổ thời gian từ dữ liệu giao dịch (point-in-time)

```python
def window_aggregates(tx: pd.DataFrame, snapshot: str, windows=(7, 30, 90)) -> pd.DataFrame:
    """tx: customer_id, ts, amount. Chỉ dùng giao dịch TRƯỚC snapshot."""
    t = pd.Timestamp(snapshot)
    tx = tx[tx["ts"] < t]
    parts = []
    for w in windows:
        sub = tx[tx["ts"] >= t - pd.Timedelta(days=w)]
        agg = sub.groupby("customer_id")["amount"].agg(["count", "sum", "mean", "max"])
        agg.columns = [f"amt_{c}_{w}d" for c in agg.columns]
        parts.append(agg)
    out = pd.concat(parts, axis=1).fillna(0)
    out["trend_7_vs_90"] = out["amt_sum_7d"] / (out["amt_sum_90d"] * 7 / 90 + 1e-9)   # tăng tốc / giảm tốc
    out["recency_days"] = (t - tx.groupby("customer_id")["ts"].max()).dt.days
    return out


tx = pd.DataFrame({"customer_id": rng.choice([f"C{i:07d}" for i in range(500)], 20_000),
                   "ts": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 200, 20_000), unit="D"),
                   "amount": rng.gamma(2, 50, 20_000).round(0)})
agg = window_aggregates(tx, "2024-06-01")
print(agg.shape)
print(agg.iloc[:3, [0, 1, 4, 5, -2, -1]].round(1))
```

Mẫu feature RFM (*Recency, Frequency, Monetary*) và *trend* (hoạt động gần đây so với dài hạn) là nhóm feature mạnh nhất cho churn trong thực tế.

### d) Feature văn bản

```python
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

tickets = ["mạng chậm vào buổi tối", "cước tháng này quá cao", "muốn hủy hợp đồng",
           "mạng rớt liên tục", "hỏi về gói cước mới", "cước cao hơn quảng cáo"]
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
M = tfidf.fit_transform(tickets)
print(M.shape, "| top features:", tfidf.get_feature_names_out()[:6])
hv = HashingVectorizer(n_features=2**10, alternate_sign=False)   # không lưu vocabulary: phù hợp streaming
print(hv.transform(tickets).shape)
```

Với tiếng Việt, nên **tách từ** trước (`underthesea`, `pyvi`) để có token như "hợp_đồng". Ở mức nâng cao, dùng embedding câu (sentence-transformers với mô hình tiếng Việt, PhoBERT) rồi giảm chiều.

### e) Quan hệ phi tuyến cho mô hình tuyến tính: spline và tương tác

**[Docs]** `SplineTransformer` tạo cơ sở B-spline, giúp mô hình tuyến tính học được đường cong mượt. `PolynomialFeatures(interaction_only=True)` tạo các tích chéo.

```python
from sklearn.preprocessing import PolynomialFeatures

base_cols = ["tenure_months", "monthly_charges", "support_calls"]
Xb = X_train[base_cols]
linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
spline = make_pipeline(SplineTransformer(n_knots=6, degree=3), LogisticRegression(max_iter=2000))
inter = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                      LogisticRegression(max_iter=2000))
for name, m in [("tuyến tính", linear), ("spline", spline), ("tương tác bậc 2", inter)]:
    s = cross_val_score(m, Xb, y_train, cv=cv, scoring="roc_auc")
    print(f"{name:16s} ROC-AUC = {s.mean():.4f}")
```

### f) Binning (rời rạc hóa)

```python
from sklearn.preprocessing import KBinsDiscretizer

kb = KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="quantile", quantile_method="averaged_inverted_cdf")
binned = kb.fit_transform(X_train[["tenure_months"]])
print("Cạnh bin (học từ train):", np.round(kb.bin_edges_[0], 1))
```

Binning làm mất thông tin, nhưng hữu ích khi cần **diễn giải** (scorecard), khi quan hệ dạng bậc thang, hoặc khi cần bền vững với ngoại lai. Binning tối ưu theo target có ở thư viện `optbinning`.

## 5.7. Pipeline tiền xử lý hoàn chỉnh

Package tham chiếu (`code/src/churn/features/build.py`) đóng gói toàn bộ thành một đối tượng:

```python norun
def build_preprocessor(scale: bool = True, random_state: int = 0) -> Pipeline:
    numeric = Pipeline([
        ("winsor", Winsorizer(0.01, 0.99)),                       # clip trước khi impute
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler() if scale else "passthrough"),  # cây không cần scale
    ])
    low_card = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False)),
    ])
    high_card = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("te", TargetEncoder(target_type="binary",
                             cv=StratifiedKFold(5, shuffle=True, random_state=random_state))),
    ])
    columns = ColumnTransformer(
        [("num", numeric, NUM_COLS), ("low", low_card, LOW_CARD_COLS),
         ("high", high_card, HIGH_CARD_COLS), ("pass", "passthrough", PASSTHROUGH)],
        remainder="drop",                 # cột lạ trong production bị bỏ qua thay vì làm hỏng mô hình
        verbose_feature_names_out=False,
    )
    return Pipeline([("domain", ChurnFeatures()), ("columns", columns)])
```

```python
prep = build_preprocessor(scale=True)
Xt = prep.fit_transform(X_train, y_train)          # TargetEncoder cần y; cross-fitting tự động
Xv = prep.transform(X_valid)
names = prep.get_feature_names_out()
print(Xt.shape, Xv.shape)
print(list(names))

clf = make_pipeline(build_preprocessor(scale=True), LogisticRegression(max_iter=3000))
res = cross_validate(clf, X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"])
print({k: round(v.mean(), 4) for k, v in res.items() if k.startswith("test_")})
```

**[Docs]** Các tính năng `ColumnTransformer` hay dùng:

- `make_column_selector(dtype_include=...)` chọn cột theo kiểu dữ liệu, không cần liệt kê tên.
- `remainder="passthrough"|"drop"|<transformer>` xử lý các cột không được liệt kê.
- `verbose_feature_names_out=False` giữ tên cột gọn (không thêm tiền tố `num__`).
- `set_output(transform="pandas")` (≥ 1.2) trả về DataFrame có tên cột, dễ debug và giải thích.
- `force_int_remainder_cols` và `n_jobs` cho bảng rất rộng.

```python
auto = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), make_column_selector(dtype_include="number")),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     make_column_selector(dtype_include=["object", "string", "category"])),
], verbose_feature_names_out=False).set_output(transform="pandas")
sample = X_train.drop(columns=["customer_id", "signup_date"]).head(1000)
print(auto.fit_transform(sample).columns[:10].tolist())
```

## 5.8. Lựa chọn đặc trưng (Feature Selection)

| Nhóm | Phương pháp (scikit-learn) | Ưu | Nhược |
|---|---|---|---|
| Filter | `VarianceThreshold`, `SelectKBest(f_classif/chi2/mutual_info_classif)`, `SelectPercentile` | Nhanh | Bỏ qua tương tác và dư thừa |
| Wrapper | `RFE`, **`RFECV`**, `SequentialFeatureSelector` | Tính đến mô hình | Chậm; dễ overfit với CV nhỏ |
| Embedded | `SelectFromModel` (L1, `feature_importances_`) | Cân bằng | Phụ thuộc mô hình; importance của cây bị bias |
| Model-agnostic | Permutation importance, SHAP, **null importance**, Boruta | Đáng tin cậy hơn | Tốn tính toán |

**Nguyên tắc vàng:** feature selection là **một bước của pipeline**. Nó phải được thực hiện **bên trong** CV, không làm trước trên toàn bộ dữ liệu (ví dụ chính thức trong *Common pitfalls*).

```python
from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest, mutual_info_classif

# Thêm 10 biến nhiễu để xem các phương pháp có loại được không
noise = pd.DataFrame(rng.normal(size=(len(Xt), 10)), columns=[f"noise_{i}" for i in range(10)])
Xsel = pd.concat([pd.DataFrame(Xt, columns=names), noise], axis=1)

# scikit-learn >= 1.8: `penalty` bị deprecate -> dùng l1_ratio=1 cho L1 (liblinear/saga hỗ trợ)
l1 = SelectFromModel(LogisticRegression(l1_ratio=1, solver="liblinear", C=0.02)).fit(Xsel, y_train)
print("L1 giữ:", [c for c, k in zip(Xsel.columns, l1.get_support()) if k])   # L1 vẫn có thể giữ vài biến nhiễu

rfecv = RFECV(LogisticRegression(max_iter=2000), step=2, cv=cv, scoring="average_precision",
              min_features_to_select=3, n_jobs=-1).fit(Xsel, y_train)
print("RFECV số feature tối ưu:", rfecv.n_features_,
      "| còn biến nhiễu:", [c for c, k in zip(Xsel.columns, rfecv.support_) if k and c.startswith("noise")])

kbest = make_pipeline(SelectKBest(mutual_info_classif, k=10), LogisticRegression(max_iter=2000))
print("SelectKBest(MI) trong pipeline, CV PR-AUC:",
      cross_val_score(kbest, Xsel, y_train, cv=cv, scoring="average_precision").mean().round(4))
```

### Null importance: phát hiện feature "ảo" của mô hình cây

```python
from lightgbm import LGBMClassifier


def null_importance(X: pd.DataFrame, y: pd.Series, n_runs: int = 15, seed: int = 0) -> pd.DataFrame:
    """So sánh importance thật với phân phối importance khi target bị xáo trộn (Altmann et al., 2010)."""
    r = np.random.default_rng(seed)
    params = dict(n_estimators=150, learning_rate=0.05, num_leaves=15, importance_type="gain", verbose=-1)
    actual = LGBMClassifier(**params, random_state=seed).fit(X, y).feature_importances_
    null = np.array([LGBMClassifier(**params, random_state=i).fit(X, r.permutation(y.to_numpy()))
                     .feature_importances_ for i in range(n_runs)])
    return pd.DataFrame({"actual": actual, "null_p90": np.percentile(null, 90, axis=0),
                         "p_value": (null >= actual).mean(axis=0)}, index=X.columns
                        ).sort_values("actual", ascending=False)


print(null_importance(Xsel, y_train).round(3).head(12))
# Feature có actual <= null_p90 (p_value lớn) không tốt hơn ngẫu nhiên -> ứng viên loại bỏ
```

## 5.9. Dữ liệu mất cân bằng lớp (Imbalanced data)

**[Docs]** imbalanced-learn, mục *Common pitfalls and recommended practices*: lỗi phổ biến nhất là **resample toàn bộ dữ liệu trước khi chia**. Lỗi này gây hai vấn đề: (1) mô hình không được đánh giá trên phân phối lớp thật; (2) mẫu tổng hợp sinh ra từ dữ liệu sẽ rơi vào tập test, gây leakage. Giải pháp là dùng **`imblearn.pipeline.Pipeline`**, trong đó bước resampling **chỉ chạy khi `fit`** và bị bỏ qua khi `predict`/`transform`.

| Kỹ thuật | Cơ chế | Khi nào dùng |
|---|---|---|
| **Không resample + metric đúng + chỉnh ngưỡng** | PR-AUC, threshold tuning (Chương 9) | **Nên thử đầu tiên** |
| Class/sample weights | Phạt lỗi lớp thiểu số nặng hơn | `class_weight="balanced"`, `scale_pos_weight` |
| Random under-sampling | Bớt lớp đa số | Dữ liệu rất lớn |
| Random over-sampling | Nhân bản lớp thiểu số | Dữ liệu nhỏ |
| SMOTE / Borderline-SMOTE / ADASYN | Nội suy mẫu thiểu số | Mô hình tuyến tính/KNN; ít giúp GBM |
| SMOTENC / SMOTEN | SMOTE cho dữ liệu có biến phân loại | |
| Ensemble resampling | `BalancedRandomForestClassifier`, `EasyEnsembleClassifier` | |
| Focal loss | Tập trung vào mẫu khó | Deep learning |

```python
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import average_precision_score, brier_score_loss

candidates = {
    "baseline": make_pipeline(build_preprocessor(), LogisticRegression(max_iter=3000)),
    "class_weight": make_pipeline(build_preprocessor(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    # imblearn >= 0.14 không cho phép Pipeline lồng làm bước trung gian -> "trải phẳng" các bước
    "SMOTE": ImbPipeline([*build_preprocessor().steps, ("smote", SMOTE(random_state=0)),
                          ("clf", LogisticRegression(max_iter=3000))]),
    "undersample": ImbPipeline([*build_preprocessor().steps, ("rus", RandomUnderSampler(random_state=0)),
                                ("clf", LogisticRegression(max_iter=3000))]),
    "BalancedRF": make_pipeline(build_preprocessor(scale=False),
                                BalancedRandomForestClassifier(n_estimators=200, sampling_strategy="all",
                                                               replacement=True, bootstrap=False,
                                                               random_state=0, n_jobs=-1)),
}
rows = []
for name, m in candidates.items():
    p = m.fit(X_train, y_train).predict_proba(X_valid)[:, 1]
    rows.append({"method": name, "pr_auc": average_precision_score(y_valid, p),
                 "brier": brier_score_loss(y_valid, p), "mean_pred": p.mean(), "true_rate": y_valid.mean()})
print(pd.DataFrame(rows).set_index("method").round(4))
```

> **Đọc kết quả:** PR-AUC (khả năng xếp hạng) gần như không đổi giữa các phương pháp. Nhưng resampling và class weight đẩy xác suất trung bình lên xa tỷ lệ thật, làm **Brier score tệ đi**, tức là **mất calibration**. Nếu xác suất được dùng trực tiếp (tính lợi nhuận kỳ vọng như Chương 1.5) thì phải calibrate lại (Chương 9.6). Trong thực tế với GBM, kết hợp `scale_pos_weight` (hoặc không gì cả) với tối ưu ngưỡng thường tốt ngang hoặc hơn SMOTE.

> **Checklist Chương 5**
> - [ ] Bước stateless trước split; mọi bước stateful nằm trong `Pipeline`, fit chỉ trên train/fold.
> - [ ] Chiến lược missing được chọn bằng CV trên metric cuối; có cờ missing khi thiếu mang tín hiệu.
> - [ ] Encoding phù hợp cardinality; `TargetEncoder` dùng `fit_transform` (cross-fitting); xử lý mức lạ/hiếm.
> - [ ] Custom transformer tuân thủ API scikit-learn (tham số trong `__init__`, thuộc tính `_`, `get_feature_names_out`) và có test.
> - [ ] Feature nghiệp vụ point-in-time, có tài liệu ý nghĩa; không dùng timestamp tuyệt đối.
> - [ ] Feature selection nằm trong CV; đã kiểm tra bằng biến nhiễu / null importance.
> - [ ] Xử lý mất cân bằng bằng `imblearn.pipeline`; đã đánh giá ảnh hưởng tới calibration.

### Tài liệu tham khảo Chương 5

- scikit-learn User Guide: *Common pitfalls and recommended practices*; *Pipelines and composite estimators* (`Pipeline`, `ColumnTransformer`, `TransformedTargetRegressor`); *Preprocessing data* (encoders, `TargetEncoder`, `KBinsDiscretizer`, `SplineTransformer`); *Imputation of missing values*; *Feature selection*; *Developing scikit-learn estimators*.
- scikit-learn Examples: *Target Encoder's Internal Cross fitting*; *Time-related feature engineering*; *Imputing missing values before building an estimator*.
- imbalanced-learn User Guide: *Over-sampling*, *Under-sampling*, *Ensemble methods*, *Common pitfalls*. imbalanced-learn.org
- Micci-Barreca, D. (2001). *A preprocessing scheme for high-cardinality categorical attributes.* ACM SIGKDD Explorations 3(1).
- Chawla, N. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR 16.
- Altmann, A. et al. (2010). *Permutation importance: a corrected feature importance measure.* Bioinformatics 26(10).
- Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection.* CRC Press (bookdown.org/max/FES).
- Zheng, A. & Casari, A. (2018). *Feature Engineering for Machine Learning.* O'Reilly.
- Google *Machine Learning Crash Course*: *Working with numerical/categorical data*. · Kaggle Learn: *Feature Engineering*.
