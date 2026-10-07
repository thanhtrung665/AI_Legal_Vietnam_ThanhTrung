# CHƯƠNG 5. TIỀN XỬ LÝ DỮ LIỆU & FEATURE ENGINEERING

## 5.1. Nguyên tắc số 1: Fit trên train, Transform trên mọi tập

Mọi phép biến đổi **học tham số từ dữ liệu** (mean để impute, min/max để scale, mapping của target encoding, vocabulary của TF-IDF...) phải được `fit` **chỉ trên tập train**, rồi `transform` cho validation/test/production. Cách an toàn nhất: gói tất cả vào **`sklearn.pipeline.Pipeline`**.

```text
Làm sạch (stateless)  →  Split  →  Pipeline[ Impute → Encode → Scale → Model ]
  (an toàn trước split)              (fit chỉ trên train, nằm trong CV)
```

| Loại bước | Ví dụ | Làm trước split được không? |
|---|---|---|
| Stateless (không học từ dữ liệu) | Sửa chính tả, chuẩn hóa chuỗi, parse ngày, tạo `tenure_years = tenure/12`, xóa trùng lặp chính xác | ✅ Được |
| Stateful (học từ dữ liệu) | Impute mean/median, scaling, encoding, feature selection, SMOTE, PCA | ❌ Phải nằm trong Pipeline |

## 5.2. Làm sạch dữ liệu (Data Cleaning)

```python
# src/churn/features/clean.py
import unicodedata

import numpy as np
import pandas as pd


def normalize_text(s: pd.Series) -> pd.Series:
    """Chuẩn hóa chuỗi: Unicode NFC, bỏ khoảng trắng thừa, chữ thường."""
    return (s.astype("string")
              .map(lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x)
              .str.strip()
              .str.replace(r"\s+", " ", regex=True)
              .str.lower()
              .astype(object)
              .where(lambda x: x.notna(), np.nan))   # pd.NA -> np.nan cho sklearn


def clean_churn(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 1. Trùng lặp: toàn dòng, sau đó theo khóa nghiệp vụ (giữ bản ghi mới nhất)
    out = out.drop_duplicates()
    out = out.sort_values("signup_date").drop_duplicates("customer_id", keep="last")

    # 2. Chuẩn hóa phân loại + gom các biến thể
    for col in ["contract", "payment_method", "region"]:
        out[col] = normalize_text(out[col])
    out["contract"] = out["contract"].replace({"monthly": "month-to-month", "1-year": "one-year"})

    # 3. Giá trị vô lý theo nghiệp vụ -> NaN (để imputer xử lý), KHÔNG tự ý xóa dòng
    out.loc[~out["age"].between(18, 100), "age"] = np.nan
    out.loc[out["monthly_charges"] <= 0, "monthly_charges"] = np.nan

    # 4. Kiểu dữ liệu
    out["signup_date"] = pd.to_datetime(out["signup_date"], errors="coerce", utc=False)
    return out.reset_index(drop=True)
```

## 5.3. Xử lý giá trị thiếu (Imputation)

| Phương pháp | Khi nào dùng | sklearn |
|---|---|---|
| Xóa dòng | Thiếu ít (<1–2%), MCAR, dữ liệu nhiều | `dropna` |
| Xóa cột | Thiếu > 60–70% và không mang tín hiệu | — |
| Mean / Median | Baseline, số; median bền với ngoại lai | `SimpleImputer(strategy="median")` |
| Mode / hằng số "missing" | Phân loại | `SimpleImputer(strategy="constant", fill_value="missing")` |
| Cờ thiếu (indicator) | Thiếu mang thông tin (MNAR) | `SimpleImputer(add_indicator=True)`, `MissingIndicator` |
| KNN | Quan hệ cục bộ, dữ liệu vừa | `KNNImputer` |
| Iterative (MICE) | MAR, quan hệ đa biến | `IterativeImputer` (experimental) |
| Forward/backward fill, nội suy | Chuỗi thời gian | `ffill`, `interpolate(method="time")` |
| Để mô hình tự xử lý | LightGBM/XGBoost/CatBoost/HistGradientBoosting xử lý NaN gốc | — |

```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor

median_imp = SimpleImputer(strategy="median", add_indicator=True)
knn_imp = KNNImputer(n_neighbors=5, weights="distance")      # nên scale trước khi dùng KNN
mice_imp = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=50, random_state=0),
    max_iter=10, random_state=0,
)
```

> **Mẹo thực tế:** So sánh các chiến lược impute **bằng cross-validation trên metric cuối cùng**, không phải bằng việc impute "giống thật" nhất.

## 5.4. Xử lý ngoại lai

| Kỹ thuật | Mô tả | Ghi chú |
|---|---|---|
| Xóa | Loại bản ghi lỗi chắc chắn | Chỉ khi xác nhận là lỗi; **không** xóa trên tập test |
| Winsorize / Clipping | Cắt về percentile [p1, p99] | Học ngưỡng trên train |
| Biến đổi | log, Box-Cox, Yeo-Johnson | Giảm ảnh hưởng đuôi dài |
| Robust scaler | Dùng median & IQR | Chương 6 |
| Mô hình bền vững | Mô hình cây, Huber loss | Ít nhạy với ngoại lai ở X |

```python
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin


class Winsorizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Cắt giá trị về [q_low, q_high] học từ tập train. Tương thích Pipeline & set_output."""

    def __init__(self, q_low: float = 0.01, q_high: float = 0.99):
        self.q_low = q_low
        self.q_high = q_high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.q_low, axis=0)
        self.upper_ = np.nanquantile(X, self.q_high, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
```

## 5.5. Mã hóa biến phân loại (Categorical Encoding)

| Encoder | Cardinality | Mô hình phù hợp | Ghi chú |
|---|---|---|---|
| One-Hot | Thấp (< 15–20) | Tuyến tính, NN | `handle_unknown="infrequent_if_exist"`, `min_frequency` để gom mức hiếm |
| Ordinal | Có thứ tự tự nhiên | Mọi mô hình | `basic < standard < premium` |
| Ordinal (tùy ý) | Bất kỳ | Mô hình cây | Cây tự tách được |
| Frequency / Count | Cao | Cây | Đơn giản, không leakage target |
| **Target / Mean encoding** | Cao | Mọi mô hình | **Phải dùng cross-fitting** để tránh leakage |
| WoE | Trung bình | Logistic (scorecard) | Diễn giải tốt |
| Hashing | Rất cao (ID, URL) | Tuyến tính | Có xung đột hash |
| Embedding | Rất cao | Neural Network | Học được biểu diễn |
| Native categorical | Bất kỳ | LightGBM, CatBoost, HistGB | CatBoost dùng ordered target statistics |

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                    sparse_output=False)

contract_order = [["month-to-month", "one-year", "two-year"]]
ordinal = OrdinalEncoder(categories=contract_order, handle_unknown="use_encoded_value",
                         unknown_value=-1)

# sklearn >= 1.3: TargetEncoder tự động cross-fitting trong fit_transform (chống leakage)
target_enc = TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=0)
```

### Frequency encoder tự viết

```python
class FrequencyEncoder(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = [X[c].value_counts(normalize=self.normalize).to_dict() for c in X.columns]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return np.column_stack([X.iloc[:, i].map(m).fillna(0).to_numpy(dtype=float)
                                for i, m in enumerate(self.maps_)])
```

## 5.6. Feature Engineering — nơi tạo ra khác biệt lớn nhất

### a) Feature nghiệp vụ (domain features)

```python
class ChurnFeatures(TransformerMixin, BaseEstimator):
    """Feature stateless dựa trên tri thức nghiệp vụ. Đặt ở đầu Pipeline."""

    def __init__(self, reference_date: str | None = None):
        self.reference_date = reference_date

    def fit(self, X: pd.DataFrame, y=None):
        # Ngày tham chiếu học 1 lần lúc fit (hoặc lấy từ config) -> train/test/production nhất quán
        self.reference_date_ = (pd.Timestamp(self.reference_date) if self.reference_date
                                else X["signup_date"].max())
        self.feature_names_out_ = np.asarray(self.transform(X).columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["days_since_signup"] = (self.reference_date_ - X["signup_date"]).dt.days
        X["signup_month"] = X["signup_date"].dt.month
        X["signup_dow"] = X["signup_date"].dt.dayofweek
        X["avg_charge_per_month"] = X["total_charges"] / X["tenure_months"].clip(lower=1)
        X["charge_vs_expected"] = X["total_charges"] / (X["monthly_charges"] * X["tenure_months"]).clip(lower=1)
        X["calls_per_year"] = X["support_calls"] / (X["tenure_months"] / 12).clip(lower=1 / 12)
        X["is_new_customer"] = (X["tenure_months"] <= 3).astype(int)
        X["high_value"] = (X["monthly_charges"] > 100).astype(int)
        return X.drop(columns=["signup_date", "customer_id"], errors="ignore")

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
```

### b) Feature thời gian — mã hóa chu kỳ

```python
def cyclical_encode(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """Tháng 12 và tháng 1 gần nhau: dùng sin/cos thay vì số nguyên."""
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df
```

### c) Feature tổng hợp từ dữ liệu giao dịch (aggregations theo cửa sổ thời gian)

```python
def window_aggregates(tx: pd.DataFrame, snapshot: str, windows=(7, 30, 90)) -> pd.DataFrame:
    """tx: customer_id, ts, amount. Chỉ dùng giao dịch TRƯỚC snapshot (point-in-time)."""
    t = pd.Timestamp(snapshot)
    tx = tx[tx["ts"] < t]
    feats = []
    for w in windows:
        sub = tx[tx["ts"] >= t - pd.Timedelta(days=w)]
        agg = sub.groupby("customer_id")["amount"].agg(["count", "sum", "mean", "std", "max"])
        agg.columns = [f"amt_{c}_{w}d" for c in agg.columns]
        feats.append(agg)
    out = pd.concat(feats, axis=1).fillna(0)
    # Feature xu hướng: hoạt động gần đây so với dài hạn
    out["trend_7_vs_90"] = out["amt_sum_7d"] / (out["amt_sum_90d"] / (90 / 7) + 1e-9)
    last = tx.groupby("customer_id")["ts"].max()
    out["recency_days"] = (t - last).dt.days
    return out
```

### d) Feature văn bản

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20_000, sublinear_tf=True)
# Tiếng Việt: tách từ trước bằng underthesea / pyvi để có "hợp_đồng", "khiếu_nại"
# Nâng cao: embeddings từ sentence-transformers (vd. "bkai-foundation-models/vietnamese-bi-encoder")
```

### e) Feature tương tác & đa thức

```python
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
spline = SplineTransformer(n_knots=5, degree=3)   # quan hệ phi tuyến mượt cho mô hình tuyến tính
```

### f) Binning (rời rạc hóa)

```python
from sklearn.preprocessing import KBinsDiscretizer

kbins = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
# strategy="kmeans" hoặc binning tối ưu theo target: thư viện `optbinning`
```

## 5.7. Pipeline tiền xử lý hoàn chỉnh (production-ready)

```python
# src/churn/features/build.py
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

NUM_COLS = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls",
            "data_usage_gb", "days_since_signup", "avg_charge_per_month",
            "charge_vs_expected", "calls_per_year"]
LOW_CARD_COLS = ["contract", "payment_method"]
HIGH_CARD_COLS = ["region"]
PASSTHROUGH = ["is_new_customer", "high_value", "signup_month", "signup_dow"]


def build_preprocessor(scale: bool = True) -> Pipeline:
    numeric = Pipeline([
        ("winsor", Winsorizer(0.01, 0.99)),          # clip trước (giữ NaN), tránh clip cột cờ 0/1
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler() if scale else "passthrough"),
    ])
    low_card = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                              sparse_output=False)),
    ])
    high_card = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("te", TargetEncoder(target_type="binary", random_state=0)),
    ])
    columns = ColumnTransformer(
        [
            ("num", numeric, NUM_COLS),
            ("low", low_card, LOW_CARD_COLS),
            ("high", high_card, HIGH_CARD_COLS),
            ("pass", "passthrough", PASSTHROUGH),
        ],
        remainder="drop",                 # an toàn: cột lạ trong production bị bỏ qua
        verbose_feature_names_out=False,
    )
    return Pipeline([("domain", ChurnFeatures()), ("columns", columns)])


preprocessor = build_preprocessor()
# X_train_t = preprocessor.fit_transform(X_train, y_train)   # TargetEncoder cần y
# X_test_t  = preprocessor.transform(X_test)
# preprocessor.get_feature_names_out()
```

> Lưu ý: trong production nên truyền `reference_date` = ngày snapshot từ config. Nếu để trống, ngày tham chiếu được học một lần lúc `fit` (ngày đăng ký lớn nhất của tập train) và dùng lại cho mọi lần `transform`.

## 5.8. Lựa chọn đặc trưng (Feature Selection)

| Nhóm | Phương pháp | Ưu | Nhược |
|---|---|---|---|
| Filter | Variance threshold, tương quan, MI, χ², IV | Nhanh | Bỏ qua tương tác |
| Wrapper | RFE/RFECV, Sequential Feature Selection | Tính đến mô hình | Chậm |
| Embedded | L1 (Lasso), tree importance | Cân bằng | Phụ thuộc mô hình |
| Model-agnostic | **Permutation importance**, SHAP, **Boruta**, null importance | Đáng tin cậy | Tốn tính toán |

```python
from sklearn.feature_selection import RFECV, SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Embedded: L1 logistic
l1_selector = SelectFromModel(
    LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced")
)

# Wrapper: RFE + CV tự chọn số lượng feature tối ưu
rfecv = RFECV(
    estimator=LogisticRegression(max_iter=2000),
    step=1, cv=StratifiedKFold(5, shuffle=True, random_state=0),
    scoring="average_precision", min_features_to_select=5, n_jobs=-1,
)
```

### Null importance — phát hiện feature "ảo"

```python
from lightgbm import LGBMClassifier


def null_importance(X: pd.DataFrame, y: pd.Series, n_runs: int = 30, seed: int = 0) -> pd.DataFrame:
    """So sánh importance thật với importance khi target bị xáo trộn."""
    rng = np.random.default_rng(seed)
    params = dict(n_estimators=200, learning_rate=0.05, verbose=-1, importance_type="gain")
    actual = LGBMClassifier(**params).fit(X, y).feature_importances_
    null = np.array([
        LGBMClassifier(**params, random_state=i).fit(X, rng.permutation(y.to_numpy())).feature_importances_
        for i in range(n_runs)
    ])
    score = np.log(1e-10 + actual / (1 + np.percentile(null, 75, axis=0)))
    return pd.DataFrame({"actual": actual, "null_p75": np.percentile(null, 75, axis=0),
                         "score": score}, index=X.columns).sort_values("score", ascending=False)
```

## 5.9. Xử lý mất cân bằng lớp (Imbalanced Data)

| Kỹ thuật | Mô tả | Khi nào dùng |
|---|---|---|
| **Không làm gì + chọn metric đúng + chỉnh ngưỡng** | PR-AUC, threshold tuning | **Mặc định nên thử đầu tiên** |
| Class weights | Phạt lỗi lớp thiểu số nặng hơn | Hầu hết mô hình hỗ trợ `class_weight` / `scale_pos_weight` |
| Random undersampling | Bớt lớp đa số | Dữ liệu rất lớn |
| Random oversampling | Nhân bản lớp thiểu số | Dữ liệu nhỏ |
| SMOTE / ADASYN / Borderline-SMOTE | Sinh mẫu tổng hợp | Mô hình tuyến tính/KNN; ít giúp GBM |
| SMOTE-NC | SMOTE cho dữ liệu hỗn hợp số + phân loại | |
| Focal loss | Tập trung vào mẫu khó | Deep learning |
| Ensemble resampling | BalancedRandomForest, EasyEnsemble | |

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline   # BẮT BUỘC dùng pipeline của imblearn

pipe = ImbPipeline([
    ("prep", build_preprocessor()),
    ("smote", SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=0)),  # chỉ chạy khi fit
    ("clf", LogisticRegression(max_iter=2000)),
])
```

> **Cảnh báo quan trọng:**
> 1. **Không bao giờ** oversample trước khi split hoặc trước cross-validation → mẫu tổng hợp rò rỉ sang tập validation → metric ảo.
> 2. Resampling/class weight **làm sai lệch xác suất** dự đoán. Nếu cần xác suất chính xác (định giá rủi ro), phải **calibrate lại** (Chương 9) hoặc hiệu chỉnh prior.
> 3. Kinh nghiệm thực tế với GBM: `scale_pos_weight` + tối ưu ngưỡng thường tốt ngang hoặc hơn SMOTE.

> **Checklist Chương 5**
> - [ ] Bước stateful nằm trong Pipeline, fit chỉ trên train.
> - [ ] Có chiến lược missing rõ ràng (kèm cờ missing khi missing mang tín hiệu).
> - [ ] Encoding phù hợp cardinality; target encoding có cross-fitting.
> - [ ] Feature engineering đảm bảo point-in-time, có tài liệu ý nghĩa nghiệp vụ.
> - [ ] Feature selection được đánh giá trong CV, không trên toàn bộ dữ liệu.
> - [ ] Xử lý mất cân bằng nằm trong pipeline; đã cân nhắc ảnh hưởng tới calibration.
