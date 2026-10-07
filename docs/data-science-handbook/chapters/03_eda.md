# CHƯƠNG 3. PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)

## 3.1. Mục tiêu của EDA

EDA không phải là "vẽ thật nhiều biểu đồ". EDA trả lời 5 nhóm câu hỏi:

1. **Cấu trúc:** bao nhiêu dòng/cột, kiểu dữ liệu, đơn vị quan sát là gì, có trùng lặp?
2. **Chất lượng:** thiếu ở đâu, vì sao thiếu, ngoại lai, giá trị vô lý, sai chuẩn hóa?
3. **Phân phối:** từng biến phân phối thế nào (lệch, đa đỉnh, đuôi dài)?
4. **Quan hệ:** biến nào liên quan đến target; các biến tương quan với nhau (đa cộng tuyến)?
5. **Rủi ro:** có dấu hiệu **leakage**, drift theo thời gian, bias theo nhóm?

> **Quy tắc quan trọng:** Làm EDA chi tiết trên **tập train** sau khi đã tách test (hoặc ít nhất là không dùng thông tin từ test để ra quyết định tiền xử lý). Nhìn vào test quá nhiều = overfitting bằng mắt.

## 3.2. Tổng quan nhanh (First look)

```python
import numpy as np
import pandas as pd

from churn.data.synthetic import make_churn_data

df = make_churn_data()

print(df.shape)
df.info(memory_usage="deep")
display(df.head())
display(df.describe(include="all").T)


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng tổng quan từng cột — thứ đầu tiên tôi chạy với mọi dataset."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=True),
        "pct_unique": (df.nunique(dropna=True) / len(df) * 100).round(2),
        "sample_values": [df[c].dropna().unique()[:5].tolist() for c in df.columns],
    }).sort_values("pct_missing", ascending=False)


display(overview(df))

# Trùng lặp: toàn dòng và theo khóa
print("Duplicated rows:", df.duplicated().sum())
print("Duplicated customer_id:", df["customer_id"].duplicated().sum())

# Phân phối target
print(df["churn"].value_counts(normalize=True).round(3))
```

### Báo cáo tự động

```python
# ydata-profiling: báo cáo HTML đầy đủ (phân phối, tương quan, missing, cảnh báo)
from ydata_profiling import ProfileReport

ProfileReport(df, title="Churn EDA", minimal=len(df) > 100_000).to_file("reports/eda.html")

# Các lựa chọn khác: sweetviz (so sánh train/test), dtale (giao diện tương tác), skimpy (terminal)
```

> Báo cáo tự động là **điểm khởi đầu**, không thay thế phân tích có giả thuyết.

## 3.3. Phân tích giá trị thiếu (Missing Values)

Ba cơ chế thiếu (Rubin, 1976) — quyết định cách xử lý:

| Cơ chế | Ý nghĩa | Ví dụ | Xử lý |
|---|---|---|---|
| **MCAR** — Missing Completely At Random | Thiếu hoàn toàn ngẫu nhiên | Lỗi truyền tin ngẫu nhiên | Xóa hoặc impute đơn giản đều không gây bias |
| **MAR** — Missing At Random | Thiếu phụ thuộc biến *quan sát được* | Người trẻ hay bỏ trống thu nhập | Impute có điều kiện (KNN, Iterative) |
| **MNAR** — Missing Not At Random | Thiếu phụ thuộc chính giá trị bị thiếu | Người thu nhập cao không khai thu nhập | Thêm cờ `is_missing`, mô hình hóa cơ chế thiếu |

```python
import matplotlib.pyplot as plt
import missingno as msno

msno.matrix(df.sample(2000, random_state=0)); plt.show()   # mẫu hình thiếu theo dòng
msno.heatmap(df); plt.show()                               # tương quan giữa các cột bị thiếu

# Thiếu có liên quan đến target không? (dấu hiệu MAR/MNAR và tín hiệu dự đoán)
for col in df.columns[df.isna().any()]:
    rate = df.groupby(df[col].isna())["churn"].mean()
    print(f"{col:15s} churn khi có={rate.get(False, np.nan):.3f} | khi thiếu={rate.get(True, np.nan):.3f}")
```

Nếu tỷ lệ churn khác biệt rõ giữa nhóm thiếu và không thiếu → **cờ missing là một feature có giá trị**.

## 3.4. Phân tích đơn biến (Univariate)

### Biến số

```python
import seaborn as sns
from scipy import stats

num_cols = ["age", "tenure_months", "monthly_charges", "total_charges",
            "support_calls", "data_usage_gb"]


def describe_numeric(s: pd.Series) -> pd.Series:
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return pd.Series({
        "mean": s.mean(), "median": s.median(), "std": s.std(),
        "skew": s.skew(), "kurtosis": s.kurt(),
        "p01": s.quantile(0.01), "p99": s.quantile(0.99),
        "n_outlier_iqr": ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum(),
        "n_zero": (s == 0).sum(),
    })


display(df[num_cols].apply(describe_numeric).T.round(2))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), num_cols):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"{col} | skew={df[col].skew():.2f}")
plt.tight_layout()
```

**Diễn giải:**

- `|skew| > 1` → lệch mạnh → cân nhắc log/Box-Cox/Yeo-Johnson (Chương 6).
- `kurtosis` cao → đuôi dày, nhiều ngoại lai.
- Nhiều giá trị 0 → có thể là "không dùng dịch vụ" (zero-inflated) hoặc giá trị mặc định thay cho missing.

### Biến phân loại

```python
cat_cols = ["contract", "payment_method", "region"]
for col in cat_cols:
    vc = df[col].value_counts(dropna=False)
    print(f"\n{col}: {df[col].nunique()} mức")
    print(pd.concat([vc, (vc / len(df) * 100).round(2)], axis=1, keys=["n", "%"]).head(10))
```

Kiểm tra: **sai chính tả/hoa thường** (`Month-to-Month` vs `month-to-month`), **mức hiếm** (< 1%), **cardinality cao** (`region` 30 mức → cần target/frequency encoding).

## 3.5. Phân tích hai biến với target (Bivariate)

```python
# Số vs target nhị phân: so sánh phân phối
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), num_cols):
    sns.boxplot(data=df, x="churn", y=col, ax=ax, showfliers=False)
plt.tight_layout()

# Phân loại vs target: tỷ lệ churn theo từng mức + khoảng tin cậy
def rate_by_category(df: pd.DataFrame, col: str, target: str = "churn") -> pd.DataFrame:
    g = df.groupby(col, observed=True)[target].agg(["mean", "count"])
    se = np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"])
    g["ci95_low"], g["ci95_high"] = g["mean"] - 1.96 * se, g["mean"] + 1.96 * se
    g["lift"] = g["mean"] / df[target].mean()
    return g.sort_values("mean", ascending=False)


display(rate_by_category(df, "contract"))

# Biến số chia bin -> tỷ lệ churn theo bin (phát hiện quan hệ phi tuyến)
df["tenure_bin"] = pd.qcut(df["tenure_months"], q=10, duplicates="drop")
df.groupby("tenure_bin", observed=True)["churn"].mean().plot(marker="o", title="Churn rate theo tenure")
```

### Weight of Evidence (WoE) & Information Value (IV)

Kỹ thuật kinh điển trong credit scoring, rất hữu ích để xếp hạng sức mạnh dự đoán của biến:

$$WoE_i = \ln\left(\frac{\%Good_i}{\%Bad_i}\right), \qquad IV = \sum_i (\%Good_i - \%Bad_i)\times WoE_i$$

```python
def information_value(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> float:
    x = df[feature]
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > bins:
        x = pd.qcut(x, q=bins, duplicates="drop")
    x = x.astype("object").where(x.notna(), "MISSING").astype(str)  # missing = 1 nhóm riêng
    tab = pd.crosstab(x, df[target])
    good = (tab[0] + 0.5) / (tab[0].sum() + 0.5)        # +0.5: làm trơn tránh log(0)
    bad = (tab[1] + 0.5) / (tab[1].sum() + 0.5)
    woe = np.log(good / bad)
    return float(((good - bad) * woe).sum())


iv = {c: information_value(df, c, "churn") for c in num_cols + cat_cols}
print(pd.Series(iv).sort_values(ascending=False).round(3))
```

| IV | Sức mạnh dự đoán |
|---|---|
| < 0.02 | Không có |
| 0.02 – 0.1 | Yếu |
| 0.1 – 0.3 | Trung bình |
| 0.3 – 0.5 | Mạnh |
| > 0.5 | **Đáng ngờ — kiểm tra leakage!** |

## 3.6. Tương quan & đa cộng tuyến

| Cặp biến | Thước đo | Ghi chú |
|---|---|---|
| Số – Số (tuyến tính) | Pearson *r* | Nhạy với ngoại lai |
| Số – Số (đơn điệu) | Spearman ρ, Kendall τ | Bền vững hơn, dùng thứ hạng |
| Phân loại – Phân loại | Cramér's V | Dựa trên χ² |
| Số – Phân loại | Correlation ratio η, ANOVA F | |
| Mọi loại, phi tuyến | Mutual Information, φK (`phik`) | Bắt quan hệ phi tuyến |

```python
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

# Spearman heatmap
corr = df[num_cols].corr(method="spearman")
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V có hiệu chỉnh bias (Bergsma, 2013)."""
    tab = pd.crosstab(x, y)
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    r, k = tab.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c, k_c = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(min(k_c - 1, r_c - 1), 1e-12)))


print("Cramér's V(contract, churn) =", round(cramers_v(df["contract"], df["churn"]), 3))

# Mutual information (bắt quan hệ phi tuyến)
X_mi = df[num_cols].fillna(df[num_cols].median())
mi = mutual_info_classif(X_mi, df["churn"], random_state=0)
print(pd.Series(mi, index=num_cols).sort_values(ascending=False).round(4))
```

### Variance Inflation Factor (VIF) — đa cộng tuyến

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_vif = add_constant(df[num_cols].dropna())
vif = pd.Series([variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
                index=X_vif.columns).drop("const")
print(vif.sort_values(ascending=False).round(2))   # VIF > 5–10: đa cộng tuyến đáng kể
```

`total_charges ≈ monthly_charges × tenure_months` → VIF cao. Với mô hình tuyến tính cần loại bỏ/kết hợp; với mô hình cây ít ảnh hưởng đến dự đoán nhưng làm **feature importance bị chia nhỏ** giữa các biến tương quan.

## 3.7. Phát hiện ngoại lai (Outliers)

```python
from sklearn.ensemble import IsolationForest


def outlier_report(s: pd.Series) -> dict:
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    z = (s - s.mean()) / s.std()
    med = s.median(); mad = (s - med).abs().median()
    robust_z = 0.6745 * (s - med) / (mad if mad else 1e-9)
    return {
        "iqr_1.5": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
        "zscore_3": int((z.abs() > 3).sum()),
        "robust_z_3.5": int((robust_z.abs() > 3.5).sum()),   # Iglewicz & Hoaglin
    }


display(pd.DataFrame({c: outlier_report(df[c]) for c in num_cols}).T)

# Ngoại lai đa biến
iso = IsolationForest(contamination=0.01, random_state=0)
flags = iso.fit_predict(df[num_cols].fillna(df[num_cols].median()))
print("Isolation Forest outliers:", (flags == -1).sum())
```

**Ngoại lai ≠ lỗi.** Câu hỏi đúng: *đây là lỗi nhập liệu, hay là sự kiện thực hiếm (VIP, gian lận)?* Hỏi domain expert trước khi xóa.

## 3.8. Phân tích theo thời gian & phát hiện drift sớm

```python
monthly = (df.set_index("signup_date")
             .resample("MS")
             .agg(n=("churn", "size"), churn_rate=("churn", "mean"),
                  avg_charge=("monthly_charges", "median")))
monthly.plot(subplots=True, figsize=(12, 7), marker="o")
```

Nếu phân phối feature hoặc tỷ lệ target thay đổi mạnh theo thời gian → bắt buộc dùng **time-based split** (Chương 7), và cần giám sát drift (Chương 12).

## 3.9. Phát hiện Leakage trong EDA

Dấu hiệu nghi ngờ:

- Một biến đơn lẻ cho AUC > 0.9 hoặc IV > 0.5.
- Biến được tạo **sau** sự kiện target (ví dụ `cancellation_reason`, `last_bill_status = "closed"`).
- ID/thời gian tuần tự tương quan với target (dữ liệu được sắp xếp theo nhãn).

```python
from sklearn.metrics import roc_auc_score


def single_feature_auc(df: pd.DataFrame, target: str) -> pd.Series:
    out = {}
    for col in df.select_dtypes("number").columns.drop(target):
        x = df[col].fillna(df[col].median())
        auc = roc_auc_score(df[target], x)
        out[col] = max(auc, 1 - auc)          # hướng không quan trọng
    return pd.Series(out).sort_values(ascending=False)


print(single_feature_auc(df, "churn").round(3))
```

### Adversarial validation — train và test có cùng phân phối?

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score


def adversarial_auc(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> float:
    """AUC ~ 0.5: cùng phân phối. AUC >> 0.5: có drift / split có vấn đề."""
    X = pd.concat([train[features], test[features]], ignore_index=True)
    y = np.r_[np.zeros(len(train)), np.ones(len(test))]
    clf = LGBMClassifier(n_estimators=200, verbose=-1)
    return cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()
```

## 3.10. Mẫu báo cáo EDA gửi stakeholder

```markdown
## Tóm tắt EDA — Churn (snapshot 2026-10)
- Dữ liệu: 20,100 dòng, 12 cột; 100 dòng trùng (đã loại). Tỷ lệ churn: 16.5%.
- Chất lượng: data_usage_gb thiếu 8% (churn khi thiếu 17.5% so với 16.5% — chênh lệch nhỏ);
  ~30 bản ghi monthly_charges gấp 8 lần bình thường, nghi lỗi nhập liệu; 50 bản ghi contract sai chuẩn hóa.
- Tín hiệu mạnh: contract (IV 0.32), tenure_months (IV 0.32), support_calls (IV 0.14).
- Rủi ro: total_charges đa cộng tuyến với tenure × monthly_charges.
- Đề xuất: time-based split; cờ missing cho data_usage_gb; target-encode region.
```

> **Checklist Chương 3**
> - [ ] Đã hiểu đơn vị quan sát, kiểm tra trùng lặp theo khóa.
> - [ ] Phân tích missing (tỷ lệ, cơ chế, quan hệ với target).
> - [ ] Phân phối đơn biến, ngoại lai đã được xác minh với domain expert.
> - [ ] Quan hệ với target (rate-by-bin, IV, MI) và đa cộng tuyến (VIF).
> - [ ] Đã kiểm tra leakage (single-feature AUC, logic thời gian) và drift theo thời gian.
> - [ ] Có báo cáo tóm tắt kèm đề xuất hành động.
