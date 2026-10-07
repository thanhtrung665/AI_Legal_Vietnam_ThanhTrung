# CHƯƠNG 6. CHUẨN HÓA, BIẾN ĐỔI PHÂN PHỐI & GIẢM CHIỀU

**Mục tiêu chương:** hiểu **khi nào** và **vì sao** cần đưa feature về cùng thang đo hoặc thay đổi hình dạng phân phối; chọn đúng scaler theo dữ liệu và mô hình; dùng giảm chiều đúng mục đích (nén feature hay trực quan hóa).

## 6.1. Vì sao thang đo quan trọng: lý thuyết

| Cơ chế | Mô hình bị ảnh hưởng | Giải thích |
|---|---|---|
| **Khoảng cách** | KNN, K-Means, SVM-RBF, DBSCAN, LOF | $\lVert x - x'\rVert^2 = \sum_j (x_j - x'_j)^2$: feature có thang lớn (VNĐ) lấn át feature thang nhỏ (số cuộc gọi) |
| **Regularization** | Ridge, Lasso, ElasticNet, Logistic (mặc định L2) | Penalty $\lambda\sum\beta_j^2$ phạt các hệ số như nhau, nhưng độ lớn của hệ số phụ thuộc đơn vị đo. Chưa scale thì regularization không công bằng |
| **Tối ưu bằng gradient** | Logistic/Linear (solver lặp), Neural Network | Mặt loss "dẹt" (điều kiện số kém) làm gradient descent hội tụ chậm hoặc dao động |
| **Phương sai** | PCA, LDA | PCA tìm hướng phương sai lớn nhất nên feature có thang lớn chiếm các thành phần đầu |
| **Thứ tự giá trị** | Cây, Random Forest, GBM | Chỉ phụ thuộc thứ tự để chọn điểm cắt, **bất biến** với mọi phép biến đổi đơn điệu |

### Thiết lập & thực nghiệm: scaling ảnh hưởng mô hình nào?

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, PowerTransformer,
                                   QuantileTransformer, RobustScaler, StandardScaler)
from sklearn.svm import SVC

from churn.data.synthetic import make_churn_data
from churn.features.clean import clean_churn

pd.set_option("display.width", 160)
df = clean_churn(make_churn_data(n=20_000, seed=42))
num_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls", "data_usage_gb"]
X = df[num_cols].fillna(df[num_cols].median()).iloc[:6000]
y = df["churn"].iloc[:6000]
cv = StratifiedKFold(5, shuffle=True, random_state=0)

models = {"KNN(k=25)": KNeighborsClassifier(25),
          "SVC-RBF": SVC(C=1.0, gamma="scale"),
          "Logistic(L2)": LogisticRegression(C=0.05, max_iter=5000),
          "HistGB (cây)": HistGradientBoostingClassifier(max_iter=100, random_state=0)}
rows = []
for name, m in models.items():
    raw = cross_val_score(m, X, y, cv=cv, scoring="roc_auc").mean()
    scaled = cross_val_score(make_pipeline(StandardScaler(), m), X, y, cv=cv, scoring="roc_auc").mean()
    rows.append({"model": name, "no_scaling": raw, "standard_scaled": scaled, "gain": scaled - raw})
print(pd.DataFrame(rows).set_index("model").round(4))
```

**Kết quả cần quan sát:** KNN và SVM cải thiện rõ khi scale. Mô hình cây cho kết quả **giống hệt**. Logistic ở đây gần như không đổi về AUC vì n lớn so với số feature nên regularization yếu. Ảnh hưởng của scaling lên regularization thể hiện ở **hệ số** (xem mục 6.2).

## 6.2. Các scaler của scikit-learn

| Scaler | Công thức | Miền | Bền ngoại lai | Giữ sparse | Dùng khi |
|---|---|---|---|---|---|
| **StandardScaler** | $z = (x-\mu)/\sigma$ | ~(−3, 3) | ❌ | Chỉ khi `with_mean=False` | Mặc định cho tuyến tính, SVM, NN, PCA |
| **MinMaxScaler** | $(x - x_{min})/(x_{max}-x_{min})$ | [0, 1] (`feature_range`) | ❌ | ❌ | NN cần miền cố định, ảnh; dùng `clip=True` cho production |
| **MaxAbsScaler** | $x / \max\lvert x\rvert$ | [−1, 1] | ❌ | ✅ | **Dữ liệu sparse** (khuyến nghị chính thức) |
| **RobustScaler** | $(x - \text{median})/\text{IQR}$ | Không cố định | ✅ | Chỉ `transform` | Nhiều ngoại lai |
| **Normalizer** | $x / \lVert x\rVert_p$ (theo **dòng**) | Vector đơn vị | — | ✅ | Văn bản TF-IDF, cosine similarity |
| **QuantileTransformer** | $F^{-1}_{target}(\hat F(x))$ | Uniform/Normal | ✅✅ | ✅ (một phần) | Ép phân phối, rất bền ngoại lai; phi tuyến |
| **PowerTransformer** | Box-Cox / Yeo-Johnson + chuẩn hóa | ~Chuẩn | Một phần | ❌ | Giảm skew, ổn định phương sai |

**[Docs]** *Scaling sparse data*: centering phá vỡ cấu trúc sparse và có thể làm tràn bộ nhớ. `MaxAbsScaler` được thiết kế cho sparse. `StandardScaler` chỉ nhận sparse khi `with_mean=False` (nếu không sẽ báo `ValueError`). `RobustScaler` không `fit` được trên sparse nhưng `transform` được.

### So sánh các scaler trên dữ liệu có ngoại lai

Mô phỏng theo ví dụ chính thức *Compare the effect of different scalers on data with outliers*:

```python
x = df[["monthly_charges"]].dropna()            # có ~30 giá trị ×8
scalers = {
    "Standard": StandardScaler(),
    "MinMax": MinMaxScaler(),
    "MaxAbs": MaxAbsScaler(),
    "Robust": RobustScaler(quantile_range=(25, 75)),
    "Quantile-Normal": QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0),
    "Yeo-Johnson": PowerTransformer(method="yeo-johnson"),
}
summary = {}
for name, s in scalers.items():
    z = s.fit_transform(x).ravel()
    body = np.quantile(z, [0.05, 0.95])
    summary[name] = {"p05": body[0], "p95": body[1], "body_width": body[1] - body[0],
                     "max": z.max(), "skew": stats.skew(z)}
print(pd.DataFrame(summary).T.round(3))
```

**Cách đọc `body_width`** (độ rộng phần thân 90% dữ liệu): với MinMax/MaxAbs, phần thân bị ép vào một khoảng rất hẹp vì một ngoại lai quyết định thang đo, làm **mất độ phân giải**. RobustScaler giữ phần thân đúng tỷ lệ nhưng ngoại lai vẫn rất lớn. Quantile-Normal kéo ngoại lai về phần đuôi của phân phối chuẩn.

### Scaling và regularization: penalty "không công bằng" khi chưa scale

Đo **mức co rút (shrinkage)** của từng hệ số: tỷ số giữa hệ số khi có regularization mạnh (C = 0.001) và hệ số gần như không regularization (C = 10⁶). Không scale thì feature có thang nhỏ (`support_calls`, 0–8) bị co mạnh, còn feature thang lớn (`monthly_charges`, `data_usage_gb`) gần như không bị phạt. Scale xong, mọi feature chịu cùng một "luật chơi".

```python
def shrinkage_table(data: pd.DataFrame, scale: bool) -> pd.Series:
    pre = [StandardScaler()] if scale else []
    strong = make_pipeline(*pre, LogisticRegression(C=1e-3, max_iter=10_000)).fit(data, y)[-1].coef_[0]
    weak = make_pipeline(*pre, LogisticRegression(C=1e6, max_iter=10_000)).fit(data, y)[-1].coef_[0]
    return pd.Series(strong / weak, index=data.columns)


X_nc = X.drop(columns="total_charges")       # bỏ biến đa cộng tuyến để so sánh co rút cho rõ
print(pd.DataFrame({"chưa scale": shrinkage_table(X_nc, scale=False),
                    "đã scale": shrinkage_table(X_nc, scale=True)}).round(3))
# Chưa scale: support_calls (thang nhỏ) bị co mạnh nhất, các biến thang lớn gần như không bị phạt
# -> mức phạt phụ thuộc đơn vị đo. Đã scale: mọi hệ số bị co theo cùng một thang (độ lệch chuẩn).
```

## 6.3. Biến đổi phân phối

### Họ biến đổi lũy thừa

**[Docs]** *Mapping to a Gaussian distribution*: power transform là họ biến đổi **đơn điệu, có tham số**, nhằm đưa dữ liệu về gần phân phối chuẩn để **ổn định phương sai và giảm skew**. λ được ước lượng bằng **maximum likelihood**.

$$\text{Box-Cox } (x>0):\; x^{(\lambda)} = \begin{cases}\dfrac{x^\lambda - 1}{\lambda} & \lambda \neq 0\\[4pt] \ln x & \lambda = 0\end{cases}$$

$$\text{Yeo-Johnson (mọi } x):\; x^{(\lambda)} = \begin{cases}[(x+1)^\lambda - 1]/\lambda & \lambda \neq 0,\ x \ge 0\\ \ln(x+1) & \lambda = 0,\ x \ge 0\\ -[(-x+1)^{2-\lambda} - 1]/(2-\lambda) & \lambda \neq 2,\ x < 0\\ -\ln(-x+1) & \lambda = 2,\ x < 0\end{cases}$$

| Biến đổi | Điều kiện | Ghi chú |
|---|---|---|
| $\log x$, $\log(1+x)$ | $x>0$ / $x \ge 0$ | Dễ diễn giải (hiệu ứng theo %); `np.log1p` |
| $\sqrt{x}$, $x^{1/3}$ | $x \ge 0$ / mọi $x$ | Dữ liệu đếm (Poisson: căn bậc hai ổn định phương sai) |
| Box-Cox | $x>0$ | λ tối ưu bằng MLE |
| **Yeo-Johnson** | Mọi $x$ | Mặc định của `PowerTransformer` |
| QuantileTransformer | Mọi $x$ | Phi tham số; làm méo khoảng cách tuyến tính; cần nhiều mẫu (`n_quantiles`) |

```python
usage = df[["data_usage_gb"]].dropna()
transforms = {
    "gốc": None,
    "log1p": lambda a: np.log1p(a),
    "box-cox": PowerTransformer(method="box-cox"),
    "yeo-johnson": PowerTransformer(method="yeo-johnson"),
    "quantile-normal": QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0),
}
for name, t in transforms.items():
    z = usage.to_numpy().ravel() if t is None else (t(usage).to_numpy().ravel() if callable(t) and not hasattr(t, "fit")
                                                     else t.fit_transform(usage).ravel())
    print(f"{name:16s} skew={stats.skew(z):7.3f}  kurtosis={stats.kurtosis(z):7.3f}")

pt = PowerTransformer(method="box-cox", standardize=False).fit(usage)
print("λ Box-Cox (MLE) =", np.round(pt.lambdas_, 3), "-> gần 0 nghĩa là log là lựa chọn gần tối ưu")
```

### Khi nào biến đổi phân phối thực sự cần?

- **Mô hình tuyến tính / GLM:** giúp quan hệ gần tuyến tính hơn và giảm ảnh hưởng ngoại lai. Cần thiết khi biến lệch mạnh.
- **Mô hình cây:** **không cần**, vì biến đổi đơn điệu không đổi thứ tự.
- **Neural network:** có ích, vì input có phân phối "đẹp" giúp tối ưu ổn định.
- **Kiểm định thống kê có giả định chuẩn:** có ích, nhưng thường nên dùng phương pháp bền vững hoặc phi tham số.

## 6.4. Biến đổi target cho hồi quy

Khi target lệch phải (doanh thu, giá nhà, thời gian xử lý), huấn luyện trên $\log y$ thường cải thiện mô hình. **[Docs]** `TransformedTargetRegressor` tự áp dụng `func` khi fit và `inverse_func` khi predict.

**Cảnh báo thống kê (bất đẳng thức Jensen):** $\exp(\mathbb{E}[\log y]) \le \mathbb{E}[y]$. Dự đoán ngược từ thang log ước lượng **trung vị**, không phải **trung bình**, nên hệ thống bị **ước lượng thấp**. Nếu metric cần trung bình (MSE, tổng doanh thu), hãy hiệu chỉnh bằng **smearing estimator** của Duan (1983): nhân với $\overline{\exp(\hat\varepsilon)}$.

```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
n = 8_000
Xr = pd.DataFrame({"tenure": rng.integers(1, 72, n), "calls": rng.poisson(1.5, n), "plan": rng.integers(0, 3, n)})
revenue = np.exp(3 + 0.02 * Xr["tenure"] + 0.3 * Xr["plan"] + rng.normal(0, 0.6, n))   # lệch phải, nhiễu nhân
Xa, Xb, ya, yb = train_test_split(Xr, revenue, test_size=0.3, random_state=0)

raw_model = HistGradientBoostingRegressor(random_state=0).fit(Xa, ya)
log_model = TransformedTargetRegressor(regressor=HistGradientBoostingRegressor(random_state=0),
                                       func=np.log, inverse_func=np.exp).fit(Xa, ya)
resid_log = np.log(ya) - np.log(log_model.predict(Xa))
smear = np.mean(np.exp(resid_log))                                  # hệ số smearing của Duan

for name, pred in [("y gốc", raw_model.predict(Xb)), ("log y", log_model.predict(Xb)),
                   ("log y + smearing", log_model.predict(Xb) * smear)]:
    print(f"{name:17s} MAE={mean_absolute_error(yb, pred):7.2f}  RMSE={mean_squared_error(yb, pred) ** 0.5:7.2f}"
          f"  tổng dự đoán/thực tế={pred.sum() / yb.sum():.3f}")
```

Mô hình trên thang log có **MAE tốt** (tối ưu trung vị) nhưng **tổng dự đoán thấp hơn thực tế**. Smearing khôi phục tổng. Điều này quan trọng khi dự báo dùng cho kế hoạch tài chính. Một lựa chọn khác là huấn luyện trực tiếp với **loss phù hợp phân phối**: `loss="poisson"`/`"gamma"` trong `HistGradientBoostingRegressor`, `objective="tweedie"` trong LightGBM (Chương 8).

## 6.5. Giảm chiều (Dimensionality Reduction)

| Phương pháp | Loại | Mục đích chính | Ghi chú (theo tài liệu) |
|---|---|---|---|
| **PCA** | Tuyến tính, không giám sát | Nén, khử tương quan, khử nhiễu | Cần scale trước; `n_components` là số nguyên hoặc tỷ lệ phương sai; `whiten=True` cho phương sai đơn vị |
| IncrementalPCA | Tuyến tính | PCA trên dữ liệu không vừa RAM | `partial_fit` theo batch |
| TruncatedSVD | Tuyến tính | Ma trận **sparse** (TF-IDF → LSA) | Không center, giữ sparse |
| KernelPCA | Phi tuyến | Cấu trúc phi tuyến | O(n²) bộ nhớ |
| NMF | Tuyến tính, không âm | Chủ đề văn bản, thành phần diễn giải được | Dữ liệu ≥ 0 |
| LDA (Linear Discriminant) | Có giám sát | Tối đa phân tách lớp | Tối đa K−1 chiều |
| **t-SNE** | Phi tuyến | **Chỉ trực quan hóa** | Không có `transform` cho dữ liệu mới; khoảng cách giữa cụm và kích thước cụm **không có nghĩa** |
| **UMAP** (`umap-learn`) | Phi tuyến | Trực quan hóa, có thể làm feature | Có `transform`; nhanh hơn t-SNE |

### PCA: toán học và thực hành

PCA tìm các hướng trực giao $w_k$ tối đa phương sai, tức là các vector riêng của ma trận hiệp phương sai, tính qua SVD của ma trận đã center: $X_c = U\Sigma V^\top$. Tỷ lệ phương sai giải thích của thành phần *k* bằng $\sigma_k^2/\sum_j \sigma_j^2$.

```python
from sklearn.decomposition import PCA

Xp = df[num_cols].dropna()
for name, pipe in [("không scale", PCA()), ("có scale", make_pipeline(StandardScaler(), PCA()))]:
    pipe.fit(Xp)
    pca = pipe if isinstance(pipe, PCA) else pipe[-1]
    print(f"{name:12s} explained variance ratio:", pca.explained_variance_ratio_.round(3))
# Không scale: PC1 ~ 100% vì total_charges (thang hàng nghìn) chiếm toàn bộ phương sai

pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=0.90, svd_solver="full")).fit(Xp)
pca = pca_pipe[-1]
print("Số thành phần giữ 90% phương sai:", pca.n_components_)
loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), index=num_cols,
                        columns=[f"PC{i + 1}" for i in range(pca.n_components_)])
print(loadings.round(2))          # loading = tương quan giữa feature (đã scale) và thành phần
```

**Ví dụ chính thức liên quan:** *Importance of Feature Scaling* của scikit-learn cho thấy PCA trên dữ liệu chưa scale bị chi phối hoàn toàn bởi feature có thang lớn nhất. Pipeline `StandardScaler → PCA → classifier` cho độ chính xác cao hơn hẳn.

### TruncatedSVD cho văn bản (LSA) và t-SNE để nhìn dữ liệu

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

docs = ["mạng chậm buổi tối", "mạng rớt liên tục", "cước tháng cao", "cước cao hơn quảng cáo",
        "muốn hủy hợp đồng", "chuyển sang nhà mạng khác", "gói cước mới", "khuyến mãi gói cước"] * 25
tfidf = TfidfVectorizer().fit_transform(docs)
lsa = TruncatedSVD(n_components=3, random_state=0).fit(tfidf)
print("LSA explained variance:", lsa.explained_variance_ratio_.round(3), "| input sparse:", tfidf.format)

sample = make_pipeline(StandardScaler()).fit_transform(Xp.sample(1500, random_state=0))
emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=0).fit_transform(sample)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(emb[:, 0], emb[:, 1], s=4, c=df.loc[Xp.sample(1500, random_state=0).index, "churn"], cmap="coolwarm")
ax.set_title("t-SNE: chỉ để nhìn, không dùng khoảng cách/kích thước cụm để kết luận")
print("t-SNE embedding:", emb.shape)
```

## 6.6. Ghép nối: scaler khác nhau cho từng nhóm cột

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

scaling = ColumnTransformer([
    ("skewed", Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("pow", PowerTransformer(method="yeo-johnson"))]), ["data_usage_gb", "total_charges"]),
    ("outliers", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("rob", RobustScaler())]), ["monthly_charges"]),
    ("regular", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("std", StandardScaler())]), ["age", "tenure_months", "support_calls"]),
], verbose_feature_names_out=False).set_output(transform="pandas")
out = scaling.fit_transform(df[num_cols])
print(out.describe().T[["mean", "std", "min", "max"]].round(2))
```

## 6.7. Pitfalls

1. **Leakage do scale trên toàn bộ dữ liệu** trước khi chia: `StandardScaler().fit(X_all)` khiến mean/std chứa thông tin tập test. Luôn đặt scaler trong `Pipeline`.
2. **Quên lưu scaler:** production phải dùng đúng scaler đã fit lúc train. Hãy lưu **cả pipeline** (`joblib`/MLflow), không lưu riêng mô hình.
3. **MinMaxScaler trong production:** giá trị mới ngoài [min, max] của train cho output ngoài [0, 1]. Dùng `clip=True` hoặc giám sát drift.
4. **Box-Cox với giá trị ≤ 0:** lỗi. Dùng Yeo-Johnson hoặc `log1p`.
5. **Scale biến one-hot:** thường không cần và làm khó diễn giải hệ số. Với L1/L2 mạnh, cân nhắc kỹ.
6. **QuantileTransformer với ít dữ liệu:** `n_quantiles` lớn hơn số mẫu sẽ bị điều chỉnh, tạo các bậc thang. Phép biến đổi phi tuyến làm méo khoảng cách nên không phù hợp mọi mô hình.
7. **Diễn giải t-SNE như phân cụm:** sai. t-SNE không bảo toàn khoảng cách toàn cục, và kết quả thay đổi theo `perplexity`.
8. **Biến đổi log target rồi báo cáo tổng/mean** mà không hiệu chỉnh Jensen.

> **Checklist Chương 6**
> - [ ] Chỉ scale khi mô hình cần (khoảng cách, regularization, gradient, PCA); mô hình cây có thể bỏ qua.
> - [ ] Chọn scaler theo phân phối và ngoại lai; dữ liệu sparse dùng `MaxAbsScaler` / `with_mean=False`.
> - [ ] Biến lệch mạnh dùng log/Yeo-Johnson cho mô hình tuyến tính/NN; kiểm tra bằng skew và Q-Q.
> - [ ] Target lệch: cân nhắc `TransformedTargetRegressor` + smearing, hoặc loss Poisson/Gamma/Tweedie.
> - [ ] PCA fit trong pipeline sau khi scale; báo cáo explained variance; t-SNE/UMAP chỉ để trực quan.
> - [ ] Scaler và mọi transformer được serialize cùng mô hình.

### Tài liệu tham khảo Chương 6

- scikit-learn User Guide: *Preprocessing data* (Standardization, Scaling sparse data, Scaling data with outliers, Non-linear transformation, Mapping to a Gaussian distribution, Normalization); *Transforming target in regression*; *Decomposing signals in components* (PCA, IncrementalPCA, KernelPCA, TruncatedSVD, NMF); *Manifold learning* (t-SNE).
- scikit-learn Examples: *Compare the effect of different scalers on data with outliers*; *Importance of Feature Scaling*; *Effect of transforming the targets in regression model*.
- Box, G. E. P. & Cox, D. R. (1964). *An analysis of transformations.* JRSS-B 26(2). · Yeo, I.-K. & Johnson, R. (2000). *A new family of power transformations.* Biometrika 87(4).
- Duan, N. (1983). *Smearing Estimate: A Nonparametric Retransformation Method.* JASA 78(383).
- Wattenberg, M., Viégas, F. & Johnson, I. (2016). *How to Use t-SNE Effectively.* Distill.
- Stanford CS229 Lecture Notes: *Principal Components Analysis*. · Hastie, Tibshirani & Friedman. *The Elements of Statistical Learning*, §3.4 (shrinkage và scaling), §14.5 (PCA).
