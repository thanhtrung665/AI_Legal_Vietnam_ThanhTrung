# CHƯƠNG 3. PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)

> *"Exploratory data analysis is detective work."* — John W. Tukey, *Exploratory Data Analysis* (1977).

**Mục tiêu chương:** hiểu cấu trúc, chất lượng, phân phối và quan hệ trong dữ liệu; phát hiện rủi ro (leakage, drift, bias) **trước khi** mô hình hóa; và kết thúc bằng các **quyết định cụ thể** cho bước tiền xử lý.

## 3.1. EDA trả lời những câu hỏi nào?

| Nhóm | Câu hỏi | Kỹ thuật | Quyết định dẫn tới |
|---|---|---|---|
| Cấu trúc | Đơn vị quan sát? khóa? trùng lặp? kiểu dữ liệu? | Profile bảng, kiểm tra khóa | Dedup, ép kiểu, định nghĩa grain |
| Chất lượng | Thiếu ở đâu, vì sao? giá trị vô lý? sai chuẩn hóa? | Missing matrix, miền giá trị | Chiến lược impute, quy tắc làm sạch |
| Phân phối | Lệch, đa đỉnh, đuôi dày, tập trung ở 0? | Histogram, ECDF, Q-Q, skew/kurtosis | Biến đổi log/Yeo-Johnson, binning |
| Quan hệ với target | Biến nào có tín hiệu, tuyến tính hay không? | Rate-by-bin, IV, MI, AUC đơn biến | Feature engineering, chọn mô hình |
| Quan hệ giữa feature | Đa cộng tuyến, dư thừa? | Spearman, Cramér's V, VIF, phân cụm feature | Loại/gộp feature, diễn giải cẩn thận |
| Thời gian | Phân phối có trôi theo thời gian? mùa vụ? | Chuỗi theo tháng, cohort | Out-of-time split, giám sát drift |
| Rủi ro | Leakage? train/test khác phân phối? | AUC đơn biến, adversarial validation | Loại feature, sửa cách chia |

**EDA và CDA.** Tukey phân biệt *Exploratory* (tìm giả thuyết) với *Confirmatory Data Analysis* (kiểm định giả thuyết, Chương 4). Một mẫu hình tìm được khi "đào bới" dữ liệu cần được **xác nhận trên dữ liệu độc lập**. Nếu không, đó có thể chỉ là kết quả của việc thử nhiều lần (*garden of forking paths*).

> **[Docs]** scikit-learn *Common pitfalls* nhắc: mọi quyết định dựa trên dữ liệu (chọn biến, ngưỡng cắt ngoại lai, cách impute) đều là một dạng "fit". Vì vậy **EDA chi tiết làm trên tập train**, sau khi đã tách test.

### Thiết lập

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from churn.data.synthetic import make_churn_data
from churn.viz.style import set_style

set_style()
raw = make_churn_data(n=20_000, seed=42)

# Chỉ dedup chính xác (stateless) trước khi tách; mọi phân tích sâu làm trên train
raw = raw.drop_duplicates().reset_index(drop=True)
train = raw[raw["signup_date"] < "2023-10-01"].copy()
test = raw[raw["signup_date"] >= "2023-10-01"].copy()
print(len(train), len(test))
```

## 3.2. Hồ sơ cấu trúc (Data profiling)

```python
def profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Bảng hồ sơ từng cột: thứ đầu tiên nên chạy với mọi dataset."""
    n = len(frame)
    top = frame.apply(lambda s: s.value_counts(dropna=True).iloc[0] / n if s.notna().any() else np.nan)
    return pd.DataFrame({
        "dtype": frame.dtypes.astype(str),
        "n_missing": frame.isna().sum(),
        "pct_missing": (frame.isna().mean() * 100).round(2),
        "n_unique": frame.nunique(dropna=True),
        "pct_unique": (frame.nunique(dropna=True) / n * 100).round(2),
        "top_freq_pct": (top * 100).round(2),      # ~100% => cột hằng/gần hằng
        "examples": [frame[c].dropna().unique()[:3].tolist() for c in frame.columns],
    })


print(profile(train).to_string())
print("Trùng toàn dòng:", train.duplicated().sum(),
      "| Trùng khóa customer_id:", train["customer_id"].duplicated().sum())
print("Tỷ lệ churn train/test:", round(train["churn"].mean(), 4), round(test["churn"].mean(), 4))
```

Cần đọc ra từ bảng này:

- `pct_unique ≈ 100%` ở cột không phải ID: có thể là timestamp hoặc số liên tục. Nếu là ID trá hình thì phải loại.
- `top_freq_pct > 95%`: cột gần như hằng số (*quasi-constant*), gần như không mang thông tin.
- Cột số mang kiểu `str`/`object`: thường do giá trị rác như `"N/A"` hay dấu phân cách hàng nghìn.

## 3.3. Giá trị thiếu: lượng, mẫu hình và cơ chế

**Ba cơ chế thiếu (Rubin, 1976)** quyết định cách xử lý:

| Cơ chế | Định nghĩa | Ví dụ | Hệ quả |
|---|---|---|---|
| **MCAR** | P(thiếu) không phụ thuộc dữ liệu nào | Lỗi đồng bộ ngẫu nhiên | Xóa dòng không gây bias, chỉ mất hiệu quả |
| **MAR** | P(thiếu) phụ thuộc biến **quan sát được** | Khách trẻ hay bỏ trống tuổi | Impute có điều kiện (KNN, MICE) |
| **MNAR** | P(thiếu) phụ thuộc **chính giá trị bị thiếu** | Người thu nhập cao không khai thu nhập | Không thể sửa chỉ bằng dữ liệu; dùng cờ missing, mô hình hóa cơ chế, thu thập thêm |

MCAR **không thể chứng minh**, chỉ có thể tìm bằng chứng chống lại. Cách thực tế: kiểm tra xem **cờ thiếu có dự đoán được từ các biến khác không**. Nếu có, dữ liệu không phải MCAR.

```python
import missingno as msno
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

msno.matrix(train.sample(1500, random_state=0), figsize=(10, 4), fontsize=9, sparkline=False)

num_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls", "data_usage_gb"]
cat_cols = ["contract", "payment_method", "region"]

# (1) Thiếu có liên quan tới target không? -> cờ missing có thể là feature
for col in ["age", "data_usage_gb", "payment_method"]:
    rates = train.groupby(train[col].isna())["churn"].agg(["mean", "size"])
    print(f"{col:15s} churn|có={rates.loc[False, 'mean']:.3f}  churn|thiếu={rates.loc[True, 'mean']:.3f}"
          f"  (n_thiếu={rates.loc[True, 'size']})")

# (2) Bằng chứng chống MCAR: dự đoán cờ thiếu từ các biến khác (AUC ≈ 0.5 -> không bác bỏ MCAR)
others = train[["tenure_months", "monthly_charges", "support_calls"]]
for col in ["age", "data_usage_gb"]:
    miss = train[col].isna().astype(int)
    p = cross_val_predict(LogisticRegression(max_iter=1000), others, miss, cv=5, method="predict_proba")[:, 1]
    print(f"AUC dự đoán cờ thiếu {col}: {roc_auc_score(miss, p):.3f}")
```

> **Diễn giải với dữ liệu mô phỏng.** Thiếu được bơm ngẫu nhiên (MCAR), nên AUC ≈ 0.5 và tỷ lệ churn hai nhóm gần nhau. Trong dữ liệu thật, AUC này thường > 0.6, tức là dữ liệu thiếu có hệ thống.

## 3.4. Phân tích đơn biến: biến số

### Thống kê mô tả cổ điển và bền vững

| Khía cạnh | Thống kê cổ điển | Thống kê bền vững (robust) |
|---|---|---|
| Vị trí | Mean | **Median**, trimmed mean |
| Độ phân tán | Std | **IQR**, **MAD** = median(\|x − median\|) |
| Hình dạng | Skewness, kurtosis | Quantile skewness (Bowley) |

Breakdown point của mean là 0%: một giá trị cực đoan đủ làm nó lệch tùy ý. Của median là 50%. Vì vậy dữ liệu có ngoại lai (như `monthly_charges`) nên được mô tả bằng thống kê bền vững.

```python
def describe_numeric(s: pd.Series) -> pd.Series:
    s = s.dropna()
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    mad = stats.median_abs_deviation(s, scale="normal")    # scale="normal": ước lượng σ khi phân phối chuẩn
    return pd.Series({
        "mean": s.mean(), "median": q2, "std": s.std(), "mad_sigma": mad, "iqr": q3 - q1,
        "skew": stats.skew(s), "kurtosis_excess": stats.kurtosis(s),
        "bowley_skew": (q3 + q1 - 2 * q2) / (q3 - q1) if q3 > q1 else 0.0,
        "p01": s.quantile(0.01), "p99": s.quantile(0.99), "max": s.max(),
        "pct_zero": (s == 0).mean() * 100,
    })


print(train[num_cols].apply(describe_numeric).T.round(2).to_string())
```

Cách đọc:

- `|skew| > 1`: lệch mạnh, nên cân nhắc log hoặc Yeo-Johnson (Chương 6). `data_usage_gb` (log-normal) là ví dụ điển hình.
- `std ≫ mad_sigma` và kurtosis rất lớn: có ngoại lai kéo std lên. Đây là trường hợp của `monthly_charges` (kurtosis ≈ 179) do các giá trị bị nhân 8.
- `pct_zero` cao: phân phối *zero-inflated*, hoặc số 0 đang đóng vai trò "không có dữ liệu".

### Hình dạng phân phối: histogram, ECDF, Q-Q

Số bin của histogram ảnh hưởng mạnh tới diễn giải. Quy tắc **Freedman–Diaconis**, độ rộng bin $h = 2\,\text{IQR}\cdot n^{-1/3}$, bền với ngoại lai (`bins="fd"` trong NumPy/matplotlib). **ECDF** không cần chọn bin và cho phép đọc trực tiếp phân vị.

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes[0], ["monthly_charges", "data_usage_gb", "tenure_months"]):
    sns.histplot(train[col].dropna(), bins="fd", ax=ax)
    ax.set_title(f"{col} | skew={stats.skew(train[col].dropna()):.2f}")
sns.ecdfplot(data=train, x="data_usage_gb", hue="churn", ax=axes[1, 0])
axes[1, 0].set_xscale("log")
stats.probplot(train["data_usage_gb"].dropna(), dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Q-Q (gốc): cong -> không chuẩn")
stats.probplot(np.log(train["data_usage_gb"].dropna()), dist="norm", plot=axes[1, 2])
axes[1, 2].set_title("Q-Q log: gần thẳng -> log-normal")
plt.tight_layout()
```

### Kiểm định phân phối chuẩn và giới hạn của nó

```python
x = np.log(train["data_usage_gb"].dropna())
print("Shapiro–Wilk (n=500) p =", round(stats.shapiro(x.sample(500, random_state=0)).pvalue, 4))
print("D'Agostino K² (toàn bộ) p =", round(stats.normaltest(x).pvalue, 4))
ad = stats.anderson(x, dist="norm")
print("Anderson–Darling A² =", round(ad.statistic, 3), "| giá trị tới hạn 5% =", ad.critical_values[2])
```

> **[Kinh nghiệm]** Với n lớn, kiểm định chuẩn **gần như luôn bác bỏ** vì độ lệch nhỏ không đáng kể vẫn có ý nghĩa thống kê. Với n nhỏ thì ngược lại, thiếu power. Hãy ra quyết định bằng **Q-Q plot + độ lớn skew/kurtosis**. p-value chỉ là thông tin phụ.

## 3.5. Phân tích đơn biến: biến phân loại

```python
for col in cat_cols:
    vc = train[col].value_counts(dropna=False)
    share = vc / len(train)
    entropy = stats.entropy(share)                      # 0 = một mức; ln(K) = đều tuyệt đối
    rare = (share < 0.01).sum()
    print(f"\n{col}: {train[col].nunique()} mức | entropy={entropy:.2f}/{np.log(len(vc)):.2f}"
          f" | mức hiếm (<1%)={rare}")
    print(pd.DataFrame({"n": vc, "%": (share * 100).round(2)}).head(6).to_string())
```

Cần phát hiện: **sai chuẩn hóa** (`Month-to-Month` và `month-to-month`), **mức hiếm** (gom thành "other" hoặc dùng `min_frequency` của `OneHotEncoder`), **cardinality cao** (`region` có 30 mức, cần target/frequency encoding).

## 3.6. Quan hệ với target

### Biến số và target nhị phân

```python
def numeric_vs_binary(frame: pd.DataFrame, cols: list[str], target: str) -> pd.DataFrame:
    rows = []
    for c in cols:
        d = frame[[c, target]].dropna()
        pos, neg = d.loc[d[target] == 1, c], d.loc[d[target] == 0, c]
        auc = roc_auc_score(d[target], d[c])
        rows.append({
            "feature": c,
            "median_pos": pos.median(), "median_neg": neg.median(),
            "ks_stat": stats.ks_2samp(pos, neg).statistic,          # khác biệt phân phối
            "auc_univariate": max(auc, 1 - auc),                    # sức xếp hạng đơn lẻ
            "direction": "+" if auc >= 0.5 else "-",
        })
    return pd.DataFrame(rows).sort_values("auc_univariate", ascending=False)


print(numeric_vs_binary(train, num_cols, "churn").round(3).to_string(index=False))
```

### Tỷ lệ target theo nhóm, có khoảng tin cậy Wilson

Khoảng tin cậy Wald $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$ cho kết quả sai khi n nhỏ hoặc p gần 0/1. **Khoảng Wilson** (mặc định nên dùng, Agresti & Coull 1998) đáng tin hơn.

```python
from statsmodels.stats.proportion import proportion_confint


def rate_by_group(frame: pd.DataFrame, col: str, target: str = "churn") -> pd.DataFrame:
    g = frame.groupby(col, observed=True, dropna=False)[target].agg(["sum", "count"])
    lo, hi = proportion_confint(g["sum"], g["count"], alpha=0.05, method="wilson")
    out = pd.DataFrame({"n": g["count"], "rate": g["sum"] / g["count"], "ci_low": lo, "ci_high": hi})
    out["lift"] = out["rate"] / frame[target].mean()
    return out.sort_values("rate", ascending=False)


print(rate_by_group(train, "contract").round(3))
print(rate_by_group(train, "payment_method").round(3))

# Biến số -> chia decile -> tỷ lệ churn (phát hiện quan hệ phi tuyến, ngưỡng)
binned = train.assign(bin=pd.qcut(train["tenure_months"], 10, duplicates="drop"))
fig, ax = plt.subplots()
r = rate_by_group(binned, "bin").sort_index()
ax.errorbar(range(len(r)), r["rate"], yerr=[r["rate"] - r["ci_low"], r["ci_high"] - r["rate"]],
            fmt="o-", capsize=3)
ax.set(title="Churn giảm đều theo tenure", xlabel="Decile tenure", ylabel="Tỷ lệ churn")
```

### Weight of Evidence & Information Value

Kỹ thuật chuẩn trong credit scoring (Siddiqi, *Intelligent Credit Scoring*):

$$\text{WoE}_i = \ln\frac{\%\text{Non-event}_i}{\%\text{Event}_i}, \qquad \text{IV} = \sum_i \left(\%\text{Non-event}_i - \%\text{Event}_i\right)\cdot \text{WoE}_i$$

```python
def woe_iv(frame: pd.DataFrame, feature: str, target: str, bins: int = 10) -> tuple[pd.DataFrame, float]:
    x = frame[feature]
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > bins:
        x = pd.qcut(x, q=bins, duplicates="drop")
    x = x.astype("object").where(x.notna(), "MISSING").astype(str)   # missing là một nhóm riêng
    tab = pd.crosstab(x, frame[target])
    good = (tab[0] + 0.5) / (tab[0].sum() + 0.5)                     # +0.5: làm trơn tránh log(0)
    bad = (tab[1] + 0.5) / (tab[1].sum() + 0.5)
    table = pd.DataFrame({"n": tab.sum(1), "event_rate": tab[1] / tab.sum(1), "woe": np.log(good / bad)})
    return table, float(((good - bad) * table["woe"]).sum())


iv = pd.Series({c: woe_iv(train, c, "churn")[1] for c in num_cols + cat_cols}).sort_values(ascending=False)
print(iv.round(3))
```

| IV | Sức mạnh dự đoán (Siddiqi) |
|---|---|
| < 0.02 | Không có |
| 0.02–0.1 | Yếu |
| 0.1–0.3 | Trung bình |
| 0.3–0.5 | Mạnh |
| > 0.5 | **Đáng ngờ, kiểm tra leakage** |

> Kết quả trên dữ liệu mô phỏng khớp với cơ chế sinh: `contract`, `tenure_months`, `support_calls` mạnh; `region` và `data_usage_gb` có IV ≈ 0, đúng với việc chúng **không có tác động thật**. Đây là cách kiểm tra EDA trên dữ liệu biết trước đáp án.

### Mutual Information: bắt quan hệ phi tuyến

**[Docs]** `mutual_info_classif` ước lượng MI bằng phương pháp k-láng giềng (Kraskov et al., 2004) cho biến liên tục. Cần khai báo `discrete_features` đúng, và kết quả có tính ngẫu nhiên nên phải đặt `random_state`.

```python
from sklearn.feature_selection import mutual_info_classif

X_mi = train[num_cols].fillna(train[num_cols].median())
mi = mutual_info_classif(X_mi, train["churn"], discrete_features=[False, True, False, False, True, False],
                         n_neighbors=3, random_state=0)
print(pd.Series(mi, index=num_cols).sort_values(ascending=False).round(4))
```

## 3.7. Quan hệ giữa các feature

| Cặp biến | Thước đo | Ghi chú |
|---|---|---|
| Số – Số, tuyến tính | Pearson *r* | Nhạy ngoại lai; chỉ đo quan hệ tuyến tính |
| Số – Số, đơn điệu | **Spearman ρ**, Kendall τ | Dùng thứ hạng, bền vững; nên dùng mặc định |
| Phân loại – Phân loại | **Cramér's V** (hiệu chỉnh bias) | Dựa trên χ² |
| Số – Phân loại | Correlation ratio η | η² = phương sai giữa nhóm / tổng phương sai |
| Mọi loại, phi tuyến | Mutual information, φK (`phik`) | |

```python
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V với hiệu chỉnh bias (Bergsma, 2013)."""
    tab = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    r, k = tab.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(min(kc - 1, rc - 1), 1e-12)))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    d = pd.DataFrame({"c": categories, "v": values}).dropna()
    grand = d["v"].mean()
    between = d.groupby("c", observed=True)["v"].agg(lambda g: len(g) * (g.mean() - grand) ** 2).sum()
    total = ((d["v"] - grand) ** 2).sum()
    return float(np.sqrt(between / total)) if total else 0.0


corr = train[num_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title("Spearman: total_charges gắn chặt với tenure")
print("Cramér's V(contract, payment_method) =", round(cramers_v(train["contract"], train["payment_method"]), 3))
print("η(contract -> monthly_charges)       =", round(correlation_ratio(train["contract"], train["monthly_charges"]), 3))
```

### Đa cộng tuyến: VIF và phân cụm feature

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

trong đó $R_j^2$ là R² khi hồi quy feature *j* theo các feature còn lại. VIF > 5–10 là đa cộng tuyến đáng kể.

```python
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_vif = add_constant(train[num_cols].dropna())
vif = pd.Series([variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
                index=X_vif.columns).drop("const")
print(vif.sort_values(ascending=False).round(2))

# [Docs] scikit-learn example "Permutation Importance with Multicollinear or Correlated Features":
# phân cụm phân cấp trên khoảng cách 1 - |Spearman|, giữ 1 feature mỗi cụm
dist = 1 - np.abs(corr.to_numpy())
np.fill_diagonal(dist, 0)
linkage = hierarchy.ward(squareform(dist, checks=False))
clusters = hierarchy.fcluster(linkage, t=0.5, criterion="distance")
print(dict(zip(num_cols, clusters)))
```

`total_charges ≈ monthly_charges × tenure_months` nên VIF cao. Với **mô hình tuyến tính**, đa cộng tuyến làm hệ số không ổn định và khó diễn giải. Với **mô hình cây**, dự đoán ít bị ảnh hưởng, nhưng **feature importance bị chia nhỏ** giữa các biến tương quan (Chương 10).

## 3.8. Ngoại lai (Outliers)

| Phương pháp | Loại | Quy tắc | Ghi chú |
|---|---|---|---|
| IQR (Tukey fences) | Đơn biến | $x < Q_1 - 1.5\,IQR$ hoặc $x > Q_3 + 1.5\,IQR$ | Không giả định phân phối |
| Z-score | Đơn biến | $\lvert z\rvert > 3$ | Mean/std bị chính ngoại lai làm sai lệch (*masking*) |
| **Robust z (MAD)** | Đơn biến | $0.6745\,\lvert x - \tilde{x}\rvert / \text{MAD} > 3.5$ | Iglewicz & Hoaglin (1993) |
| Mahalanobis bền vững | Đa biến | Khoảng cách với hiệp phương sai MCD | Giả định elip |
| Isolation Forest | Đa biến | Điểm dễ bị cô lập bằng cây ngẫu nhiên | Không giả định phân phối, nhanh |
| Local Outlier Factor | Đa biến | Mật độ cục bộ thấp hơn láng giềng | Tốt khi có nhiều cụm |

```python
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def univariate_outliers(s: pd.Series) -> dict:
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    med, mad = s.median(), stats.median_abs_deviation(s)
    return {
        "iqr_1.5": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
        "z_3": int((np.abs(stats.zscore(s)) > 3).sum()),
        "robust_z_3.5": int((0.6745 * np.abs(s - med) / (mad or 1e-9) > 3.5).sum()),
    }


print(pd.DataFrame({c: univariate_outliers(train[c]) for c in num_cols}).T)

Xo = train[["monthly_charges", "tenure_months", "support_calls"]].dropna()
mcd = MinCovDet(random_state=0).fit(Xo)
d2 = mcd.mahalanobis(Xo)                                   # bình phương khoảng cách Mahalanobis bền vững
flag_mcd = d2 > stats.chi2.ppf(0.999, df=Xo.shape[1])      # ngưỡng χ² với 3 bậc tự do
# contamination="auto" dùng ngưỡng của bài báo gốc -> thường gắn cờ quá nhiều; đặt theo tỷ lệ kỳ vọng
flag_if = IsolationForest(contamination=0.005, random_state=0).fit_predict(Xo) == -1
flag_lof = LocalOutlierFactor(n_neighbors=35).fit_predict(Xo) == -1
print({"mcd": int(flag_mcd.sum()), "isolation_forest": int(flag_if.sum()), "lof": int(flag_lof.sum())})
print("Ngoại lai cài sẵn (>500) trong train:", int((Xo["monthly_charges"] > 500).sum()),
      "| bị MCD bắt:", int((flag_mcd & (Xo["monthly_charges"] > 500).to_numpy()).sum()))
```

**Quy trình ra quyết định với ngoại lai:**

1. **Có thể xảy ra về mặt vật lý/nghiệp vụ không?** Không (tuổi 250) thì là lỗi, chuyển thành NaN.
2. **Có thể xảy ra nhưng hiếm?** Hỏi domain expert. Đó có thể là khách VIP hoặc gian lận, tức là **tín hiệu quý**.
3. **Mô hình có nhạy với ngoại lai không?** Mô hình tuyến tính/KNN thì clip (winsorize) hoặc biến đổi. Mô hình cây ít nhạy với ngoại lai ở X.
4. **Không bao giờ** xóa ngoại lai trên tập test để "làm đẹp" kết quả.

## 3.9. Chiều thời gian

```python
by_month = (raw.set_index("signup_date").resample("MS")
               .agg(n=("churn", "size"), churn_rate=("churn", "mean"),
                    median_charges=("monthly_charges", "median"),
                    pct_m2m=("contract", lambda s: (s.str.lower() == "month-to-month").mean())))
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for ax, col in zip(axes, ["churn_rate", "median_charges", "pct_m2m"]):
    ax.plot(by_month.index, by_month[col], marker="o", ms=3)
    ax.set_ylabel(col)
axes[0].set_title("Theo dõi target và feature theo thời gian: nền tảng cho out-of-time validation")
plt.tight_layout()
print(by_month[["churn_rate"]].describe().T.round(3))
```

Nếu tỷ lệ target hoặc phân phối feature thay đổi rõ theo thời gian, **bắt buộc** chia dữ liệu theo thời gian (Chương 7) và thiết lập giám sát drift (Chương 12).

## 3.10. Phát hiện leakage ngay từ EDA

Dấu hiệu nghi ngờ:

- Một feature đơn lẻ có AUC > 0.9 hoặc IV > 0.5.
- Feature được ghi nhận **sau** sự kiện target: `cancellation_reason`, `refund_amount`, `last_status = "closed"`.
- ID, số thứ tự hoặc timestamp tương quan với target (dữ liệu được sắp xếp theo nhãn).

Minh họa: thêm một feature rò rỉ `days_to_contract_end_at_export`, được tính tại thời điểm **xuất dữ liệu** (sau khi khách đã hủy).

```python
leaky = train.copy()
rng = np.random.default_rng(0)
# Khách đã hủy có "số ngày còn lại của hợp đồng" = 0 vì hệ thống đóng hợp đồng khi hủy
leaky["days_to_contract_end_at_export"] = np.where(leaky["churn"] == 1, 0, rng.integers(1, 365, len(leaky)))
leaky["row_number"] = np.arange(len(leaky))

single_auc = numeric_vs_binary(leaky, num_cols + ["days_to_contract_end_at_export", "row_number"], "churn")
print(single_auc[["feature", "auc_univariate"]].head(4).round(3).to_string(index=False))
# AUC = 1.0 -> không có thật ở thời điểm dự đoán: loại bỏ và truy nguyên cách feature được tạo
```

## 3.11. Adversarial validation: train và test có cùng phân phối?

Huấn luyện một bộ phân loại phân biệt **dòng thuộc train** với **dòng thuộc test**. AUC ≈ 0.5 nghĩa là hai tập giống nhau. AUC càng cao, phân phối càng khác. Feature importance của bộ phân loại chỉ ra **feature nào trôi**.

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold


def adversarial_validation(a: pd.DataFrame, b: pd.DataFrame, features: list[str]) -> tuple[float, pd.Series]:
    X = pd.concat([a[features], b[features]], ignore_index=True)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category")
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, importance_type="gain",
                         verbose=-1, random_state=0)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    imp = pd.Series(clf.fit(X, y).feature_importances_, index=features).sort_values(ascending=False)
    return roc_auc_score(y, p), imp


feats = ["age", "tenure_months", "monthly_charges", "support_calls", "data_usage_gb", "contract", "payment_method"]
perm = np.random.default_rng(1).permutation(len(train))           # hai nửa RỜI NHAU của train
half_a, half_b = train.iloc[perm[: len(train) // 2]], train.iloc[perm[len(train) // 2:]]
auc_same, _ = adversarial_validation(half_a, half_b, feats)
with_time = [*feats, "signup_ordinal"]
tr_t = train.assign(signup_ordinal=train["signup_date"].map(pd.Timestamp.toordinal))
te_t = test.assign(signup_ordinal=test["signup_date"].map(pd.Timestamp.toordinal))
auc_time, imp = adversarial_validation(tr_t, te_t, with_time)
print(f"AUC (hai nửa ngẫu nhiên của train) = {auc_same:.3f}")
print(f"AUC (train vs test theo thời gian, có ngày đăng ký) = {auc_time:.3f} | feature trôi nhất: {imp.index[0]}")
```

Feature trôi theo thời gian (như ngày đăng ký tuyệt đối) cho AUC ≈ 1. Đó là lý do **không đưa timestamp thô vào mô hình**. Thay vào đó dùng feature tương đối, đo tới **thời điểm chấm điểm của từng dòng**, như `tenure_months` (xem bẫy "ngày tham chiếu cố định" ở Chương 5.6).

## 3.12. Báo cáo tự động: điểm khởi đầu, không phải điểm kết thúc

```python norun
from ydata_profiling import ProfileReport

ProfileReport(train, title="Churn EDA", minimal=len(train) > 100_000,
              explorative=True).to_file("reports/eda_profile.html")
# sweetviz.compare([train, "train"], [test, "test"], target_feat="churn"): so sánh 2 tập
```

## 3.13. Mẫu báo cáo EDA gửi stakeholder

```markdown
## Tóm tắt EDA: Churn (dữ liệu train đến 2023-09)
**Dữ liệu:** ~13 000 khách, 12 cột; 100 dòng trùng chính xác (đã loại). Tỷ lệ churn train 16–17%.
**Chất lượng:**
- data_usage_gb thiếu 8%, age 5%, payment_method 3%; không có bằng chứng thiếu có hệ thống (AUC ≈ 0.5).
- Một số bản ghi monthly_charges gấp ~8 lần bình thường (lỗi nhập liệu, ~30 trên toàn bộ dữ liệu): winsorize.
- Một số bản ghi contract sai chuẩn hóa ("Month-to-Month"): chuẩn hóa chữ thường.
**Tín hiệu:** contract (IV≈0.3), tenure_months (IV≈0.3), support_calls (IV≈0.15).
  region và data_usage_gb gần như không có tín hiệu.
**Rủi ro:** total_charges đa cộng tuyến với tenure × monthly_charges; ngày đăng ký trôi theo thời gian.
**Đề xuất:** out-of-time split; winsorize + median impute; one-hot cho contract/payment;
  target encoding có cross-fitting cho region; không dùng timestamp thô làm feature.
```

> **Checklist Chương 3**
> - [ ] EDA chi tiết làm trên tập train; test chỉ dùng cho adversarial validation.
> - [ ] Đã kiểm tra khóa, trùng lặp, kiểu dữ liệu, cột hằng/gần hằng.
> - [ ] Missing: lượng, mẫu hình, quan hệ với target, bằng chứng về cơ chế.
> - [ ] Phân phối mô tả bằng thống kê bền vững; quyết định biến đổi dựa trên Q-Q và skew.
> - [ ] Quan hệ với target (rate + CI, IV, MI, AUC đơn biến) và giữa các feature (Spearman, Cramér's V, VIF).
> - [ ] Ngoại lai được phân loại: lỗi hay hiện tượng thật (đã hỏi domain expert).
> - [ ] Đã kiểm tra drift theo thời gian, leakage, adversarial validation.
> - [ ] Có báo cáo tóm tắt kèm quyết định cho bước tiền xử lý.

### Tài liệu tham khảo Chương 3

- Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley.
- Rubin, D. B. (1976). *Inference and missing data.* Biometrika 63(3). · van Buuren, S. (2018). *Flexible Imputation of Missing Data*, 2nd ed. (stefvanbuuren.name/fimd)
- scikit-learn User Guide: *Mutual information*, *Novelty and Outlier Detection*, *Covariance estimation*; Example: *Permutation Importance with Multicollinear or Correlated Features*.
- SciPy reference: `scipy.stats` (shapiro, normaltest, anderson, ks_2samp, median_abs_deviation).
- statsmodels: `proportion_confint`, `variance_inflation_factor`.
- Agresti, A. & Coull, B. (1998). *Approximate is better than "exact" for interval estimation of binomial proportions.* The American Statistician 52(2).
- Iglewicz, B. & Hoaglin, D. (1993). *How to Detect and Handle Outliers.* ASQC.
- Siddiqi, N. (2017). *Intelligent Credit Scoring*, 2nd ed. Wiley.
- Harvard CS109 *Data Science*, các bài EDA & Visualization. · Kaggle Learn: *Data Leakage*.
