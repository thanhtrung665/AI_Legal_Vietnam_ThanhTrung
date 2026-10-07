# CHƯƠNG 6. CHUẨN HÓA & BIẾN ĐỔI DỮ LIỆU (SCALING & TRANSFORMATION)

## 6.1. Khi nào cần chuẩn hóa?

| Nhóm mô hình | Cần scale? | Lý do |
|---|---|---|
| Dựa trên khoảng cách: KNN, K-Means, SVM (RBF), DBSCAN | ✅ **Bắt buộc** | Biến có thang đo lớn chi phối khoảng cách |
| Tối ưu bằng gradient: Linear/Logistic (có regularization), Neural Network | ✅ **Bắt buộc** | Hội tụ nhanh hơn; penalty L1/L2 công bằng giữa các hệ số |
| PCA, LDA | ✅ Bắt buộc | PCA tối đa phương sai → biến thang lớn chiếm ưu thế |
| Mô hình cây: Decision Tree, Random Forest, GBM | ❌ Không cần | Chỉ dựa trên thứ tự giá trị (split thresholds) |
| Naive Bayes | Tùy loại | Gaussian NB không cần, nhưng cần phân phối gần chuẩn |

## 6.2. Các phương pháp scaling

| Phương pháp | Công thức | Miền giá trị | Bền với ngoại lai | Dùng khi |
|---|---|---|---|---|
| **StandardScaler** (Z-score) | $z = \dfrac{x - \mu}{\sigma}$ | ~(-3, 3) | ❌ | Mặc định cho tuyến tính, NN, SVM |
| **MinMaxScaler** | $x' = \dfrac{x - x_{min}}{x_{max} - x_{min}}$ | [0, 1] | ❌ | NN với sigmoid, ảnh (pixel), cần miền cố định |
| **RobustScaler** | $x' = \dfrac{x - \text{median}}{IQR}$ | Không cố định | ✅ | Dữ liệu nhiều ngoại lai |
| **MaxAbsScaler** | $x' = \dfrac{x}{\max\lvert x\rvert}$ | [-1, 1] | ❌ | Dữ liệu thưa (sparse) — giữ số 0 |
| **Normalizer** (theo dòng) | $x' = \dfrac{x}{\lVert x \rVert_2}$ | Vector đơn vị | — | Văn bản TF-IDF, cosine similarity |

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler,
                                   StandardScaler)

from churn.data.synthetic import make_churn_data

df = make_churn_data()
x = df[["monthly_charges"]].dropna()          # có ~30 ngoại lai x8

scalers = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler(quantile_range=(25, 75)),
    "maxabs": MaxAbsScaler(),
}
summary = {name: pd.Series(s.fit_transform(x).ravel()).describe()[["mean", "std", "min", "50%", "max"]]
           for name, s in scalers.items()}
print(pd.DataFrame(summary).round(3))
# MinMax: phần lớn dữ liệu bị ép vào [0, 0.15] vì 1 ngoại lai -> mất phân giải
# Robust: phần thân phân phối giữ nguyên tỷ lệ
```

## 6.3. Biến đổi phân phối (làm "chuẩn hóa" hình dạng)

| Phương pháp | Điều kiện | Ghi chú |
|---|---|---|
| Log: $\log(x)$, $\log(1+x)$ | $x > 0$ / $x \ge 0$ | Đơn giản, dễ diễn giải; `np.log1p` |
| Căn bậc hai, căn bậc ba | $x \ge 0$ / mọi $x$ | Lệch vừa phải; dữ liệu đếm |
| **Box-Cox** | $x > 0$ | Tự tìm λ tối ưu (MLE) |
| **Yeo-Johnson** | Mọi $x$ (kể cả âm, 0) | Mặc định của `PowerTransformer` |
| **QuantileTransformer** | Mọi $x$ | Ép về Uniform/Normal; phi tuyến mạnh, bền với ngoại lai; cần nhiều dữ liệu |
| Rank / percentile | Mọi $x$ | Rank Gauss trong các cuộc thi |

$$\text{Box-Cox: } x^{(\lambda)} = \begin{cases} \dfrac{x^\lambda - 1}{\lambda} & \lambda \neq 0 \\ \ln x & \lambda = 0 \end{cases}$$

```python
from scipy import stats
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, QuantileTransformer

usage = df[["data_usage_gb"]].dropna()

transforms = {
    "raw": FunctionTransformer(),
    "log1p": FunctionTransformer(np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one"),
    "box-cox": PowerTransformer(method="box-cox"),          # yêu cầu > 0
    "yeo-johnson": PowerTransformer(method="yeo-johnson"),
    "quantile-normal": QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                           random_state=0),
}
for name, t in transforms.items():
    z = t.fit_transform(usage).ravel()
    print(f"{name:16s} skew={stats.skew(z):6.3f}  kurtosis={stats.kurtosis(z):6.3f}")

pt = PowerTransformer(method="box-cox").fit(usage)
print("λ tối ưu =", pt.lambdas_)
```

### Biến đổi target cho hồi quy

Khi target lệch phải (doanh thu, giá nhà), huấn luyện trên $\log(y)$ thường tốt hơn; `TransformedTargetRegressor` tự động biến đổi ngược khi dự đoán:

```python
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor

reg = TransformedTargetRegressor(
    regressor=LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1),
    func=np.log1p, inverse_func=np.expm1,
)
# Lưu ý: expm1(E[log y]) ≈ median chứ không phải mean của y (Jensen) -> có thể cần hiệu chỉnh smearing
```

## 6.4. Giảm chiều (Dimensionality Reduction)

| Phương pháp | Loại | Dùng cho | Ghi chú |
|---|---|---|---|
| PCA | Tuyến tính | Nén feature, khử tương quan | Cần scale trước |
| TruncatedSVD | Tuyến tính | Ma trận thưa (TF-IDF) | LSA |
| LDA (Linear Discriminant) | Có giám sát | Tối đa phân tách lớp | ≤ K-1 chiều |
| t-SNE | Phi tuyến | **Chỉ trực quan hóa** | Không bảo toàn khoảng cách toàn cục |
| UMAP | Phi tuyến | Trực quan hóa + làm feature | Nhanh hơn t-SNE, `umap-learn` |
| Autoencoder | Phi tuyến | Dữ liệu lớn/phức tạp | Deep learning |

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

num = df[["age", "tenure_months", "monthly_charges", "total_charges",
          "support_calls", "data_usage_gb"]].dropna()

pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver="full"))
Z = pca_pipe.fit_transform(num)
pca = pca_pipe.named_steps["pca"]
print("Số thành phần giữ 95% phương sai:", pca.n_components_)
print("Explained variance ratio:", pca.explained_variance_ratio_.round(3))

loadings = pd.DataFrame(pca.components_.T, index=num.columns,
                        columns=[f"PC{i + 1}" for i in range(pca.n_components_)])
print(loadings.round(2))
```

## 6.5. Ghép nối: scaling khác nhau cho từng nhóm cột

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

skewed = ["data_usage_gb", "total_charges"]
heavy_outlier = ["monthly_charges"]
regular = ["age", "tenure_months", "support_calls"]

scaling = ColumnTransformer([
    ("skewed", Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("pow", PowerTransformer(method="yeo-johnson"))]), skewed),
    ("robust", Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("rob", RobustScaler())]), heavy_outlier),
    ("std", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("std", StandardScaler())]), regular),
]).set_output(transform="pandas")      # giữ tên cột -> dễ debug, dễ giải thích
```

## 6.6. Pitfalls

1. **Data leakage do scale trên toàn bộ dữ liệu:** `StandardScaler().fit(X)` trước khi split → mean/std chứa thông tin tập test. Luôn đặt scaler trong Pipeline.
2. **Scale biến one-hot:** thường không cần; với mô hình có regularization, có thể giữ 0/1 để dễ diễn giải.
3. **Scale target phân loại:** không bao giờ.
4. **Box-Cox với giá trị ≤ 0:** lỗi — dùng Yeo-Johnson.
5. **Quên lưu scaler:** production phải dùng *đúng* scaler đã fit lúc train → lưu cả pipeline (`joblib`), không lưu riêng model.
6. **Thay đổi phân phối trong production:** min/max của MinMaxScaler cố định từ train → giá trị mới có thể nằm ngoài [0, 1]. Cần `clip=True` hoặc giám sát drift.

> **Checklist Chương 6**
> - [ ] Chỉ scale khi mô hình cần; mô hình cây có thể bỏ qua.
> - [ ] Chọn scaler theo phân phối & ngoại lai; biến lệch dùng log/Yeo-Johnson.
> - [ ] Scaler nằm trong Pipeline, được serialize cùng mô hình.
> - [ ] PCA/giảm chiều được fit chỉ trên train và đã kiểm tra explained variance.
