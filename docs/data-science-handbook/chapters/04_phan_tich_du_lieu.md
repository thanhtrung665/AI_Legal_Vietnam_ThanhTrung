# CHƯƠNG 4. PHÂN TÍCH DỮ LIỆU & THỐNG KÊ SUY LUẬN

EDA giúp *nhìn thấy* mẫu hình; phân tích thống kê giúp *khẳng định* mẫu hình đó không phải do ngẫu nhiên, và *định lượng* mức độ chắc chắn. Đây là kỹ năng phân biệt Data Scientist với người chỉ "chạy mô hình".

## 4.1. Bốn cấp độ phân tích

| Cấp độ | Câu hỏi | Kỹ thuật |
|---|---|---|
| Mô tả (Descriptive) | Chuyện gì đã xảy ra? | Thống kê mô tả, KPI, dashboard, cohort |
| Chẩn đoán (Diagnostic) | Tại sao xảy ra? | Kiểm định giả thuyết, phân tích phân khúc, drill-down, hồi quy |
| Dự đoán (Predictive) | Chuyện gì sẽ xảy ra? | Machine Learning, dự báo chuỗi thời gian |
| Đề xuất (Prescriptive) | Nên làm gì? | Tối ưu hóa, uplift modeling, A/B test, mô phỏng |

## 4.2. Thống kê mô tả theo nhóm & phân tích cohort

```python
import numpy as np
import pandas as pd

from churn.data.synthetic import make_churn_data

df = make_churn_data().drop_duplicates()
df["contract"] = df["contract"].str.lower()

# Pivot nhiều chiều
pivot = pd.pivot_table(df, index="contract", columns="payment_method", values="churn",
                       aggfunc="mean", margins=True).round(3)

# Cohort theo tháng đăng ký: tỷ lệ churn & doanh thu trung bình
df["cohort"] = df["signup_date"].dt.to_period("Q")
cohort = df.groupby("cohort").agg(
    customers=("customer_id", "nunique"),
    churn_rate=("churn", "mean"),
    arpu=("monthly_charges", "median"),
)
```

Với dữ liệu sự kiện (event log), bảng cohort retention kinh điển:

```python
def retention_matrix(events: pd.DataFrame) -> pd.DataFrame:
    """events: customer_id, event_date. Trả về % khách còn hoạt động sau k tháng."""
    e = events.copy()
    e["month"] = e["event_date"].dt.to_period("M")
    e["cohort"] = e.groupby("customer_id")["month"].transform("min")
    e["age"] = (e["month"] - e["cohort"]).apply(lambda p: p.n)
    counts = e.groupby(["cohort", "age"])["customer_id"].nunique().unstack(fill_value=0)
    return counts.div(counts[0], axis=0).round(3)
```

## 4.3. Kiểm định giả thuyết — khung tư duy

1. Phát biểu **H₀** (không có khác biệt) và **H₁**.
2. Chọn mức ý nghĩa α (thường 0.05) **trước khi** nhìn dữ liệu.
3. Chọn kiểm định phù hợp (bảng dưới), kiểm tra giả định.
4. Tính p-value **và effect size + khoảng tin cậy** (p-value nhỏ ≠ khác biệt có ý nghĩa thực tế).
5. Hiệu chỉnh khi kiểm định nhiều lần (Bonferroni, Benjamini–Hochberg).

### Bảng chọn kiểm định

| Mục đích | Dữ liệu | Tham số (giả định chuẩn) | Phi tham số |
|---|---|---|---|
| So sánh 2 nhóm độc lập | Số | Welch's t-test | Mann–Whitney U |
| So sánh 2 nhóm ghép cặp | Số | Paired t-test | Wilcoxon signed-rank |
| So sánh ≥ 3 nhóm | Số | One-way ANOVA (Welch ANOVA) | Kruskal–Wallis |
| Độc lập giữa 2 biến phân loại | Tần số | χ² test of independence | Fisher's exact (mẫu nhỏ) |
| So sánh 2 tỷ lệ | Nhị phân | z-test cho tỷ lệ | Fisher's exact |
| Kiểm tra phân phối chuẩn | Số | Shapiro–Wilk (n<5000), D'Agostino | Q-Q plot |
| So sánh 2 phân phối | Số | — | Kolmogorov–Smirnov |
| Tương quan | Số | Pearson | Spearman, Kendall |

```python
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

churn_yes = df.loc[df["churn"] == 1, "monthly_charges"].dropna()
churn_no = df.loc[df["churn"] == 0, "monthly_charges"].dropna()

# 1) Welch t-test (không giả định phương sai bằng nhau — nên dùng mặc định)
t, p = stats.ttest_ind(churn_yes, churn_no, equal_var=False)

# 2) Mann–Whitney U (phi tham số, bền vững với ngoại lai)
u, p_mw = stats.mannwhitneyu(churn_yes, churn_no, alternative="two-sided")


# 3) Effect size: Cohen's d và rank-biserial correlation
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled


d = cohens_d(churn_yes.to_numpy(), churn_no.to_numpy())
rank_biserial = 1 - 2 * u / (len(churn_yes) * len(churn_no))
print(f"Welch p={p:.2e}, MWU p={p_mw:.2e}, Cohen's d={d:.3f}, r_rb={rank_biserial:.3f}")
# |d|: 0.2 nhỏ, 0.5 trung bình, 0.8 lớn

# 4) Chi-square: contract có liên quan đến churn?
table = pd.crosstab(df["contract"], df["churn"])
chi2, p_chi, dof, expected = stats.chi2_contingency(table)
assert (expected >= 5).all(), "Kỳ vọng < 5 -> dùng Fisher's exact hoặc gộp nhóm"

# 5) Kruskal–Wallis: data_usage khác nhau giữa các loại hợp đồng?
groups = [g["data_usage_gb"].dropna() for _, g in df.groupby("contract")]
h, p_kw = stats.kruskal(*groups)

# 6) So sánh 2 tỷ lệ + khoảng tin cậy Wilson
cash = df["payment_method"] == "cash"
count = np.array([df.loc[cash, "churn"].sum(), df.loc[~cash, "churn"].sum()])
nobs = np.array([cash.sum(), (~cash).sum()])
z, p_prop = proportions_ztest(count, nobs)
ci_cash = proportion_confint(count[0], nobs[0], method="wilson")
```

### Hiệu chỉnh đa kiểm định

Kiểm định 20 giả thuyết với α = 0.05 → kỳ vọng ~1 kết quả "có ý nghĩa" giả.

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.001, 0.01, 0.02, 0.04, 0.2, 0.5]
reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")  # Benjamini–Hochberg
```

## 4.4. Bootstrap — khoảng tin cậy cho mọi thống kê

Khi không có công thức giải tích (median, AUC, chênh lệch tỷ lệ phức tạp), bootstrap là công cụ vạn năng.

```python
def bootstrap_ci(data, stat_fn, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    stats_ = np.array([stat_fn(rng.choice(data, size=len(data), replace=True))
                       for _ in range(n_boot)])
    return np.quantile(stats_, [alpha / 2, 1 - alpha / 2])


print("95% CI median charges:", bootstrap_ci(churn_yes, np.median))

# scipy >= 1.7 có sẵn (phương pháp BCa chính xác hơn percentile)
res = stats.bootstrap((churn_yes.to_numpy(),), np.median, n_resamples=5000,
                      method="BCa", random_state=0)
print(res.confidence_interval)
```

## 4.5. A/B Testing chuẩn production

### Bước 1 — Tính cỡ mẫu (power analysis) TRƯỚC khi chạy

```python
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

baseline, mde = 0.18, 0.02            # churn 18%, muốn phát hiện giảm 2 điểm %
effect = proportion_effectsize(baseline, baseline - mde)
n_per_group = NormalIndPower().solve_power(effect_size=effect, alpha=0.05, power=0.8,
                                           ratio=1.0, alternative="two-sided")
print(f"Cần ~{int(np.ceil(n_per_group)):,} khách mỗi nhóm")
```

### Bước 2 — Kiểm tra Sample Ratio Mismatch (SRM)

```python
def srm_check(n_control: int, n_treatment: int, expected_ratio: float = 0.5) -> float:
    total = n_control + n_treatment
    expected = [total * (1 - expected_ratio), total * expected_ratio]
    return stats.chisquare([n_control, n_treatment], f_exp=expected).pvalue


# p < 0.001 -> phân bổ ngẫu nhiên có lỗi -> KHÔNG tin kết quả thí nghiệm
```

### Bước 3 — Phân tích kết quả

```python
def analyze_ab(conv_c: int, n_c: int, conv_t: int, n_t: int, alpha: float = 0.05) -> dict:
    p_c, p_t = conv_c / n_c, conv_t / n_t
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    _, p_value = proportions_ztest([conv_t, conv_c], [n_t, n_c])
    return {
        "control": p_c, "treatment": p_t,
        "abs_diff": diff, "rel_lift": diff / p_c,
        "ci95": (diff - z_crit * se, diff + z_crit * se),
        "p_value": p_value,
    }
```

**Kỹ thuật nâng cao:**

- **CUPED** (Controlled-experiment Using Pre-Experiment Data): giảm phương sai 20–50% bằng covariate trước thí nghiệm.
- **Sequential testing / mSPRT:** cho phép "nhìn trộm" kết quả mà không lạm phát sai lầm loại I.
- **Bayesian A/B:** trả lời trực tiếp "xác suất B tốt hơn A là bao nhiêu".

```python
# CUPED: Y_adj = Y - θ (X - mean(X)), θ = cov(X, Y) / var(X)
def cuped(y: np.ndarray, x_pre: np.ndarray) -> np.ndarray:
    theta = np.cov(x_pre, y)[0, 1] / np.var(x_pre, ddof=1)
    return y - theta * (x_pre - x_pre.mean())


# Bayesian A/B với Beta-Binomial
def prob_b_better(conv_a, n_a, conv_b, n_b, draws=200_000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.beta(1 + conv_a, 1 + n_a - conv_a, draws)
    b = rng.beta(1 + conv_b, 1 + n_b - conv_b, draws)
    return (b > a).mean(), np.mean(b / a - 1)  # P(B>A), expected relative lift
```

## 4.6. Hồi quy thống kê để giải thích (statsmodels)

Mô hình ML tối ưu dự đoán; hồi quy thống kê tối ưu **diễn giải** (hệ số, p-value, CI).

```python
import statsmodels.formula.api as smf

model = smf.logit(
    "churn ~ C(contract, Treatment('two-year')) + tenure_months + monthly_charges"
    " + support_calls + C(payment_method)",
    data=df.dropna(subset=["payment_method"]),
).fit(disp=False)
print(model.summary())

odds_ratios = pd.DataFrame({
    "OR": np.exp(model.params),
    "CI_low": np.exp(model.conf_int()[0]),
    "CI_high": np.exp(model.conf_int()[1]),
    "p": model.pvalues,
}).round(3)
# OR(support_calls)=1.42 => mỗi cuộc gọi hỗ trợ thêm, odds churn tăng 42% (giữ các biến khác cố định)
```

> **Tương quan ≠ nhân quả.** Hệ số hồi quy chỉ là nhân quả khi không có biến gây nhiễu (confounder) bị bỏ sót. Muốn kết luận nhân quả: A/B test, hoặc các phương pháp quasi-experiment (Difference-in-Differences, Propensity Score Matching, Regression Discontinuity, Instrumental Variables) — thư viện `DoWhy`, `EconML`, `CausalML`.

## 4.7. Phân tích chuỗi thời gian cơ bản

```python
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

daily = df.set_index("signup_date").resample("D")["customer_id"].count()

# Phân rã xu hướng - mùa vụ - phần dư
stl = STL(daily, period=7, robust=True).fit()
stl.plot()

# Kiểm định tính dừng: ADF (H0: không dừng), KPSS (H0: dừng) — dùng cả hai
print("ADF p =", adfuller(daily.dropna())[1])
print("KPSS p =", kpss(daily.dropna(), regression="c", nlags="auto")[1])
```

## 4.8. Nghịch lý Simpson — bài học kinh điển

Tỷ lệ churn của chương trình khuyến mãi A có thể *thấp hơn* B ở **mọi** phân khúc, nhưng *cao hơn* khi gộp chung — vì A được áp dụng nhiều hơn cho phân khúc rủi ro cao. Luôn phân tích theo phân khúc (stratify) với các biến gây nhiễu quan trọng.

```python
def simpson_check(df: pd.DataFrame, treatment: str, outcome: str, stratum: str) -> pd.DataFrame:
    overall = df.groupby(treatment)[outcome].mean().rename("overall")
    by_stratum = df.groupby([stratum, treatment])[outcome].mean().unstack()
    return by_stratum, overall
```

> **Checklist Chương 4**
> - [ ] Kết luận dựa trên kiểm định phù hợp, đã kiểm tra giả định.
> - [ ] Báo cáo effect size + khoảng tin cậy, không chỉ p-value.
> - [ ] Hiệu chỉnh đa kiểm định khi kiểm định nhiều giả thuyết.
> - [ ] A/B test: tính cỡ mẫu trước, kiểm tra SRM, không dừng sớm tùy tiện.
> - [ ] Phân biệt rõ tương quan và nhân quả; kiểm tra Simpson's paradox.
