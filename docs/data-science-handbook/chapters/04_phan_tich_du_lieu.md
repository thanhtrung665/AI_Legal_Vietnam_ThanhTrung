# CHƯƠNG 4. PHÂN TÍCH DỮ LIỆU & THỐNG KÊ SUY LUẬN

**Mục tiêu chương:** EDA giúp *nhìn thấy* mẫu hình. Thống kê suy luận giúp *khẳng định* mẫu hình đó không do ngẫu nhiên, *định lượng* độ chắc chắn và, khi có thiết kế phù hợp, *trả lời câu hỏi nhân quả*. Đây là kỹ năng phân biệt Data Scientist với người chỉ "chạy mô hình".

## 4.1. Bốn cấp độ phân tích

| Cấp độ | Câu hỏi | Kỹ thuật | Chương |
|---|---|---|---|
| Mô tả (Descriptive) | Chuyện gì đã xảy ra? | Thống kê mô tả, KPI, cohort | 3, 4.2 |
| Chẩn đoán (Diagnostic) | Tại sao xảy ra? | Kiểm định, hồi quy, phân tích phân khúc | 4.3–4.7 |
| Dự đoán (Predictive) | Chuyện gì sẽ xảy ra? | Machine Learning, chuỗi thời gian | 8 |
| Đề xuất (Prescriptive) | Nên làm gì? | A/B test, suy luận nhân quả, uplift, tối ưu hóa | 1.7, 4.6, 4.8 |

### Thiết lập

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from churn.data.synthetic import make_churn_data
from churn.features.clean import clean_churn

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 12)
df = clean_churn(make_churn_data(n=20_000, seed=42))
rng = np.random.default_rng(2026)
print(df.shape, round(df["churn"].mean(), 4))
```

## 4.2. Nền tảng: phân phối mẫu, sai số chuẩn, khoảng tin cậy

- **Tham số (parameter)** là đại lượng của quần thể, ví dụ tỷ lệ churn thật p. **Thống kê (statistic)** tính từ mẫu, ví dụ $\hat p$.
- **Phân phối mẫu (sampling distribution)** là phân phối của thống kê qua các lần lấy mẫu lặp lại. Độ lệch chuẩn của nó là **sai số chuẩn (SE)**. Với trung bình: $SE = \sigma/\sqrt{n}$.
- **Định lý giới hạn trung tâm (CLT):** với n đủ lớn, trung bình mẫu xấp xỉ phân phối chuẩn **bất kể phân phối gốc**, miễn là phương sai hữu hạn. Phân phối càng lệch thì càng cần n lớn.

```python
population = rng.lognormal(mean=2.5, sigma=0.8, size=1_000_000)       # rất lệch phải (data usage)
for n in [5, 30, 200]:
    means = rng.choice(population, size=(5_000, n)).mean(axis=1)
    print(f"n={n:3d}  skew(mean)={stats.skew(means):5.2f}  SE thực nghiệm={means.std():.3f}"
          f"  SE lý thuyết={population.std() / np.sqrt(n):.3f}")
```

**Diễn giải đúng khoảng tin cậy 95% (tần suất):** nếu lặp lại quy trình lấy mẫu và tính khoảng nhiều lần, **95% các khoảng** sẽ chứa tham số thật. Câu "tham số có 95% xác suất nằm trong khoảng này" là sai theo trường phái tần suất. Đó là cách diễn giải của *khoảng khả tín (credible interval)* Bayes.

```python
true_p, n, covered = 0.165, 400, 0
for _ in range(2_000):
    sample = rng.random(n) < true_p
    p_hat = sample.mean()
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    covered += (p_hat - 1.96 * se <= true_p <= p_hat + 1.96 * se)
print("Tỷ lệ khoảng Wald 95% chứa p thật:", covered / 2_000)
```

### p-value: định nghĩa và sáu nguyên tắc của ASA (2016)

**p-value** là xác suất, **giả sử H₀ đúng**, quan sát được thống kê kiểm định cực đoan bằng hoặc hơn giá trị thực tế. Tuyên bố của Hiệp hội Thống kê Hoa Kỳ (Wasserstein & Lazar, 2016):

1. p-value cho biết dữ liệu **không tương thích** với một mô hình thống kê cụ thể đến mức nào.
2. p-value **không** đo xác suất giả thuyết đúng, cũng không đo xác suất dữ liệu do ngẫu nhiên tạo ra.
3. Kết luận khoa học và quyết định kinh doanh **không nên** chỉ dựa trên việc p có vượt ngưỡng hay không.
4. Suy luận đúng đòi hỏi **báo cáo đầy đủ và minh bạch**. Không được chọn lọc kết quả (p-hacking).
5. p-value **không** đo độ lớn hiệu ứng hay tầm quan trọng của kết quả.
6. Tự thân p-value không phải là thước đo bằng chứng tốt cho một mô hình hay giả thuyết.

| | H₀ đúng | H₀ sai |
|---|---|---|
| **Bác bỏ H₀** | Sai lầm loại I (α), "false positive" | Đúng. **Power** = 1 − β |
| **Không bác bỏ** | Đúng | Sai lầm loại II (β), "false negative" |

## 4.3. Chọn kiểm định và kiểm tra giả định

| Mục đích | Dữ liệu | Tham số | Phi tham số / thay thế |
|---|---|---|---|
| 2 nhóm độc lập | Số | **Welch's t-test** (mặc định, không giả định phương sai bằng nhau) | Mann–Whitney U |
| 2 nhóm ghép cặp | Số | Paired t-test | Wilcoxon signed-rank |
| ≥ 3 nhóm | Số | One-way ANOVA / **Welch ANOVA** | Kruskal–Wallis |
| 2 biến phân loại | Bảng tần số | χ² test of independence | Fisher exact (ô kỳ vọng < 5) |
| 2 tỷ lệ | Nhị phân | z-test cho tỷ lệ | Fisher exact, Barnard |
| Ghép cặp nhị phân | Trước/sau | — | McNemar |
| Phân phối chuẩn? | Số | Shapiro–Wilk, D'Agostino | Q-Q plot |
| 2 phân phối giống nhau? | Số | — | Kolmogorov–Smirnov, Anderson–Darling k-sample |
| Phương sai bằng nhau? | Số | Bartlett (nhạy với phi chuẩn) | **Levene / Brown–Forsythe** |
| Tương quan | Số | Pearson | Spearman, Kendall |

> **Lưu ý về Mann–Whitney U.** Kiểm định này **không phải** "t-test cho median". Nó kiểm định $P(X > Y) = 0.5$ (stochastic equality). Chỉ khi hai phân phối có cùng hình dạng thì nó mới trở thành kiểm định về vị trí (median).

```python
churn_yes = df.loc[df["churn"] == 1, "monthly_charges"].dropna().to_numpy()
churn_no = df.loc[df["churn"] == 0, "monthly_charges"].dropna().to_numpy()

print("Levene (phương sai bằng nhau?) p =", f"{stats.levene(churn_yes, churn_no, center='median').pvalue:.3g}")
welch = stats.ttest_ind(churn_yes, churn_no, equal_var=False)
ci = welch.confidence_interval(confidence_level=0.95)                 # CI cho hiệu hai trung bình
mw = stats.mannwhitneyu(churn_yes, churn_no, alternative="two-sided")
print(f"Welch t={welch.statistic:.2f} p={welch.pvalue:.2e} | Δmean 95% CI=({ci.low:.2f}, {ci.high:.2f})")
print(f"Mann–Whitney U p={mw.pvalue:.2e}")
```

### Effect size: độ lớn hiệu ứng (bắt buộc báo cáo cùng p-value)

| Bối cảnh | Effect size | Diễn giải quy ước (Cohen, 1988) |
|---|---|---|
| Hai trung bình | Cohen's d, **Hedges' g** (hiệu chỉnh mẫu nhỏ) | 0.2 nhỏ · 0.5 vừa · 0.8 lớn |
| Hai phân phối (phi tham số) | Rank-biserial r = 2·AUC − 1 | 0.1 nhỏ · 0.3 vừa · 0.5 lớn |
| Hai tỷ lệ | Chênh lệch tuyệt đối, **relative risk**, odds ratio | Theo ngữ cảnh kinh doanh |
| Bảng phân loại | Cramér's V | Phụ thuộc bậc tự do |
| ANOVA | η², ω² (ít bias hơn) | 0.01 · 0.06 · 0.14 |

```python
def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / pooled
    return d * (1 - 3 / (4 * (na + nb) - 9))            # hiệu chỉnh mẫu nhỏ


rank_biserial = 1 - 2 * mw.statistic / (len(churn_yes) * len(churn_no))
print(f"Hedges' g = {hedges_g(churn_yes, churn_no):.3f} | rank-biserial = {abs(rank_biserial):.3f}")
# p rất nhỏ nhưng g ~ 0.2: khác biệt CÓ THẬT nhưng NHỎ -> p-value không nói về độ lớn
```

### Bảng phân loại và tỷ lệ

```python
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

table = pd.crosstab(df["contract"], df["churn"])
chi2, p_chi, dof, expected = stats.chi2_contingency(table)
print(f"χ²={chi2:.1f}, dof={dof}, p={p_chi:.2e}; ô kỳ vọng nhỏ nhất = {expected.min():.1f}")

cash = (df["payment_method"] == "cash").to_numpy()
count = np.array([df.loc[cash, "churn"].sum(), df.loc[~cash, "churn"].sum()])
nobs = np.array([cash.sum(), (~cash).sum()])
z, p_prop = proportions_ztest(count, nobs)
lo, hi = confint_proportions_2indep(count[0], nobs[0], count[1], nobs[1], method="newcomb")
rr = (count[0] / nobs[0]) / (count[1] / nobs[1])
print(f"cash vs khác: Δp 95% CI=({lo:.3f}, {hi:.3f}), RR={rr:.2f}, p={p_prop:.2e}")

kw = stats.kruskal(*[g["support_calls"].to_numpy() for _, g in df.groupby("contract")])
print("Kruskal–Wallis support_calls ~ contract: p =", round(kw.pvalue, 3))
```

## 4.4. Kiểm định nhiều giả thuyết

Kiểm định 20 giả thuyết đúng-H₀ với α = 0.05 thì xác suất có **ít nhất một** "phát hiện" giả là 1 − 0.95²⁰ ≈ 64%.

| Mục tiêu kiểm soát | Phương pháp | Khi nào dùng |
|---|---|---|
| **FWER**: P(≥ 1 sai lầm loại I) | Bonferroni (α/m), **Holm** (luôn mạnh hơn Bonferroni) | Ít giả thuyết, sai lầm rất đắt (y tế, pháp lý) |
| **FDR**: tỷ lệ kỳ vọng phát hiện sai | **Benjamini–Hochberg**, Benjamini–Yekutieli (phụ thuộc tùy ý) | Sàng lọc nhiều feature/phân khúc |

```python
from statsmodels.stats.multitest import multipletests

# Mô phỏng: 200 phân khúc, chỉ 10 phân khúc có hiệu ứng thật (p nhỏ), 190 phân khúc H0 đúng
p_values = np.r_[rng.uniform(0, 0.0004, 10), rng.uniform(0, 1, 190)]
is_true = np.r_[np.ones(10, bool), np.zeros(190, bool)]
print(f"{'none':10s}: phát hiện {(p_values < 0.05).sum():3d} | sai {((p_values < 0.05) & ~is_true).sum()}")
for method in ["bonferroni", "holm", "fdr_bh"]:
    reject = multipletests(p_values, alpha=0.05, method=method)[0]
    print(f"{method:10s}: phát hiện {reject.sum():3d} | sai {(reject & ~is_true).sum()} | đúng {(reject & is_true).sum()}/10")
```

## 4.5. Phương pháp lấy mẫu lại: bootstrap và kiểm định hoán vị

**Bootstrap** (Efron, 1979) ước lượng phân phối mẫu của *bất kỳ* thống kê nào bằng cách lấy mẫu có hoàn lại từ dữ liệu. **[Docs]** `scipy.stats.bootstrap` mặc định dùng phương pháp **BCa** (bias-corrected and accelerated), chính xác hơn phương pháp percentile khi phân phối lệch.

```python
res = stats.bootstrap((churn_yes,), np.median, n_resamples=5_000, method="BCa", rng=rng)
print("95% CI BCa cho median cước (khách churn):", np.round(res.confidence_interval, 2))


def median_diff(a, b, axis=-1):
    return np.median(a, axis=axis) - np.median(b, axis=axis)


# Hai mẫu lớn: BCa cần jackknife O(n) lần tính thống kê -> rất chậm; percentile đủ tốt khi n lớn
res2 = stats.bootstrap((churn_yes, churn_no), median_diff, n_resamples=2_000, method="percentile", rng=rng)
print("95% CI chênh lệch median:", np.round(res2.confidence_interval, 2))
```

**Kiểm định hoán vị (permutation test)** kiểm định H₀ "hai nhóm có cùng phân phối" một cách chính xác, không cần giả định phân phối: xáo trộn nhãn nhóm nhiều lần để tạo phân phối null.

```python
def mean_diff(a, b, axis=-1):
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)


small_a, small_b = churn_yes[:60], churn_no[:80]          # mẫu nhỏ: minh họa kiểm định chính xác
perm = stats.permutation_test((small_a, small_b), mean_diff, n_resamples=9_999,
                              alternative="two-sided", rng=rng)
print(f"Permutation test: Δmean={perm.statistic:.2f}, p={perm.pvalue:.4f}")
```

## 4.6. A/B testing chuẩn ngành

**[Sách]** Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments* (2020). Đây là chuẩn mực được Microsoft, Google, Booking.com, LinkedIn áp dụng.

### Thiết kế

| Thành phần | Quyết định | Ví dụ Churn |
|---|---|---|
| **OEC** (Overall Evaluation Criterion) | Một metric chính phản ánh giá trị dài hạn | Tỷ lệ giữ chân 60 ngày |
| Đơn vị ngẫu nhiên hóa | Người dùng, phiên, cụm (địa bàn) | Khách hàng |
| Guardrail metrics | Không được xấu đi | Doanh thu/khách, khiếu nại, hủy do bị làm phiền |
| MDE (Minimum Detectable Effect) | Hiệu ứng nhỏ nhất *có ý nghĩa kinh doanh* | Giảm churn 2 điểm % |
| α, power | Thường 0.05 và 0.8 | |
| Thời lượng | Đủ cỡ mẫu **và** đủ chu kỳ tuần | ≥ 2 tuần đầy đủ |

### Bước 1: tính cỡ mẫu trước khi chạy

$$n \approx \frac{2\,(z_{1-\alpha/2} + z_{1-\beta})^2\,\bar p(1-\bar p)}{\delta^2}$$

```python
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

baseline, mde = 0.18, 0.02
effect = proportion_effectsize(baseline, baseline - mde)          # Cohen's h
n_per_group = NormalIndPower().solve_power(effect_size=effect, alpha=0.05, power=0.8, ratio=1.0)
print(f"Cần ~{int(np.ceil(n_per_group)):,} khách mỗi nhóm để phát hiện giảm {mde:.0%} từ {baseline:.0%}")
for m in [0.01, 0.02, 0.04]:
    n_m = NormalIndPower().solve_power(proportion_effectsize(baseline, baseline - m), alpha=0.05, power=0.8)
    print(f"  MDE={m:.0%} -> n/nhóm={int(np.ceil(n_m)):,}")    # MDE giảm 1/2 -> n tăng ~4 lần
```

### Bước 2: kiểm tra tính hợp lệ (trustworthiness)

- **A/A test:** chạy hai nhóm giống hệt nhau. Tỷ lệ "có ý nghĩa" phải xấp xỉ α. Nếu không, hạ tầng thí nghiệm có lỗi.
- **Sample Ratio Mismatch (SRM):** tỷ lệ phân bổ thực tế lệch khỏi thiết kế là dấu hiệu lỗi nghiêm trọng (bot, lỗi redirect, lỗi log). **Không** phân tích tiếp khi có SRM.

```python
def srm_pvalue(n_control: int, n_treatment: int, expected_ratio: float = 0.5) -> float:
    total = n_control + n_treatment
    return stats.chisquare([n_control, n_treatment],
                           f_exp=[total * (1 - expected_ratio), total * expected_ratio]).pvalue


print("SRM p (50 000 vs 50 300):", round(srm_pvalue(50_000, 50_300), 4))   # ổn
print("SRM p (50 000 vs 51 200):", f"{srm_pvalue(50_000, 51_200):.2e}")    # p < 0.001 -> dừng, điều tra
```

### Bước 3: vấn đề "nhìn trộm" (peeking)

Xem kết quả mỗi ngày và dừng ngay khi p < 0.05 sẽ **lạm phát sai lầm loại I** rất nhiều. Mô phỏng A/A test (không có hiệu ứng thật):

```python
def peeking_false_positive_rate(n_sims: int = 1_000, n_days: int = 20, users_per_day: int = 500) -> float:
    false_pos = 0
    for _ in range(n_sims):
        a = rng.random((n_days, users_per_day)) < 0.18
        b = rng.random((n_days, users_per_day)) < 0.18           # cùng tỷ lệ: H0 đúng
        ca, cb = a.sum(axis=1).cumsum(), b.sum(axis=1).cumsum()
        n = users_per_day * np.arange(1, n_days + 1)
        pooled = (ca + cb) / (2 * n)
        z = (ca / n - cb / n) / np.sqrt(pooled * (1 - pooled) * 2 / n)
        false_pos += np.any(np.abs(z) > 1.96)                    # dừng ngay khi "có ý nghĩa"
    return false_pos / n_sims


print("Tỷ lệ false positive khi nhìn trộm 20 lần:", peeking_false_positive_rate())   # >> 0.05
```

Giải pháp: cố định cỡ mẫu và chỉ phân tích một lần; hoặc dùng **sequential testing** có kiểm soát α (alpha spending O'Brien–Fleming, mSPRT / always-valid p-values). Các nền tảng Optimizely, Eppo, Statsig đều có sẵn những phương pháp này.

### Bước 4: phân tích, với CUPED để giảm phương sai

**CUPED** (Deng et al., 2013, Microsoft) dùng covariate **trước thí nghiệm** X (không bị can thiệp ảnh hưởng) để giảm phương sai:

$$Y^{cuped} = Y - \theta\,(X - \bar X),\qquad \theta = \frac{\operatorname{cov}(X, Y)}{\operatorname{var}(X)},\qquad \operatorname{Var}(Y^{cuped}) = \operatorname{Var}(Y)(1-\rho^2)$$

```python
n_exp = 8_000
pre_usage = rng.gamma(4, 5, 2 * n_exp)                              # sử dụng trước thí nghiệm
group = np.r_[np.zeros(n_exp), np.ones(n_exp)]
post_usage = 0.8 * pre_usage + rng.normal(0, 4, 2 * n_exp) + 0.6 * group   # hiệu ứng thật = +0.6

theta = np.cov(pre_usage, post_usage)[0, 1] / np.var(pre_usage, ddof=1)
y_cuped = post_usage - theta * (pre_usage - pre_usage.mean())
for name, y in [("raw", post_usage), ("CUPED", y_cuped)]:
    t = stats.ttest_ind(y[group == 1], y[group == 0], equal_var=False)
    ci = t.confidence_interval()
    print(f"{name:6s} Δ={y[group == 1].mean() - y[group == 0].mean():.3f} "
          f"CI=({ci.low:.3f}, {ci.high:.3f}) p={t.pvalue:.2e}")
```

### Phân tích Bayes (Beta–Binomial)

```python
def bayes_ab(conv_a: int, n_a: int, conv_b: int, n_b: int, draws: int = 200_000) -> dict:
    a = rng.beta(1 + conv_a, 1 + n_a - conv_a, draws)            # prior Beta(1,1)
    b = rng.beta(1 + conv_b, 1 + n_b - conv_b, draws)
    lift = b / a - 1
    return {"P(B<A)": float((b < a).mean()),                      # churn: B tốt hơn nếu thấp hơn
            "expected_rel_change": float(lift.mean()),
            "95% credible": np.round(np.quantile(lift, [0.025, 0.975]), 3).tolist()}


print(bayes_ab(conv_a=1_800, n_a=10_000, conv_b=1_650, n_b=10_000))
```

> **Metric dạng tỷ số** (doanh thu/phiên, CTR theo trang) có đơn vị phân tích khác đơn vị ngẫu nhiên hóa. Khi đó phương sai phải tính bằng **delta method** hoặc bootstrap theo người dùng. Dùng công thức nhị thức thông thường sẽ cho CI quá hẹp.

## 4.7. Hồi quy để suy luận (statsmodels)

Mô hình ML tối ưu **dự đoán**. Hồi quy thống kê tối ưu **diễn giải**: hệ số, sai số chuẩn, CI, kiểm định.

### Hồi quy logistic: odds ratio và marginal effects

```python
d = df.dropna(subset=["payment_method", "age"]).copy()
logit = smf.logit(
    "churn ~ C(contract, Treatment('two-year')) + tenure_months + monthly_charges"
    " + support_calls + C(payment_method, Treatment('credit-card')) + age",
    data=d,
).fit(disp=False)

or_table = pd.DataFrame({"OR": np.exp(logit.params), "CI_low": np.exp(logit.conf_int()[0]),
                         "CI_high": np.exp(logit.conf_int()[1]), "p": logit.pvalues}).round(3)
print(or_table)
print("Pseudo R² (McFadden):", round(logit.prsquared, 4))

# Average marginal effects: thay đổi XÁC SUẤT (không phải odds) khi biến tăng 1 đơn vị
print(logit.get_margeff(at="overall").summary_frame().round(4).head(6))
```

> Đối chiếu với cơ chế sinh dữ liệu (Chương 0.8): hệ số log-odds thật của `support_calls` là 0.35, tương ứng OR = e^0.35 ≈ 1.42. Ước lượng hồi quy phải bao khoảng giá trị này. Đây là cách kiểm tra mô hình thống kê "đúng đặc tả".

### Hồi quy tuyến tính và chẩn đoán phần dư

```python
from statsmodels.stats.diagnostic import het_breuschpagan

ols = smf.ols("np.log(data_usage_gb) ~ age + tenure_months + C(contract)", data=d.dropna()).fit()
bp = het_breuschpagan(ols.resid, ols.model.exog)
print(f"Breusch–Pagan p={bp[1]:.3f} (p nhỏ -> phương sai sai số không đều)")
robust = ols.get_robustcov_results(cov_type="HC3")          # sai số chuẩn bền với heteroscedasticity
print(pd.DataFrame({"coef": ols.params, "se_classic": ols.bse, "se_HC3": robust.bse}).round(4))
```

| Giả định OLS | Kiểm tra | Hậu quả khi vi phạm | Khắc phục |
|---|---|---|---|
| Tuyến tính | Residual vs fitted, partial residual plot | Hệ số sai lệch | Biến đổi, spline, tương tác |
| Sai số độc lập | Durbin–Watson, cấu trúc nhóm/thời gian | SE quá nhỏ | Cluster-robust SE, mô hình hỗn hợp |
| Phương sai đều | Breusch–Pagan, White | SE sai | **HC3 robust SE**, WLS |
| Sai số chuẩn | Q-Q residual | Ảnh hưởng ít khi n lớn | Bootstrap |
| Không đa cộng tuyến hoàn hảo | VIF | Hệ số không ổn định | Bỏ/gộp biến, ridge |

### GLM cho dữ liệu đếm (Poisson)

```python
poisson = smf.glm("support_calls ~ C(contract) + tenure_months", data=d,
                  family=sm.families.Poisson()).fit()
dispersion = poisson.pearson_chi2 / poisson.df_resid
print("Incidence rate ratios:", np.exp(poisson.params).round(3).to_dict())
print("Hệ số phân tán:", round(dispersion, 2), "(>>1 -> overdispersion, dùng Negative Binomial)")
```

## 4.8. Suy luận nhân quả: nhập môn có thực hành

> **Tương quan ≠ nhân quả.** Hệ số hồi quy chỉ có nghĩa nhân quả khi không còn **biến gây nhiễu (confounder)** bị bỏ sót, tức là giả định *no unmeasured confounding*.

### Nghịch lý Simpson

```python
# Dữ liệu quan sát: chương trình khuyến mãi được áp dụng nhiều cho nhóm RỦI RO CAO
n_obs = 20_000
risk_high = rng.random(n_obs) < 0.5
promo = rng.random(n_obs) < np.where(risk_high, 0.8, 0.2)              # gây nhiễu: risk -> promo
p_churn = np.where(risk_high, 0.40, 0.10) - 0.05 * promo                # promo GIẢM churn 5 điểm %
y_obs = rng.random(n_obs) < p_churn
sim = pd.DataFrame({"risk_high": risk_high, "promo": promo, "churn": y_obs})

print("Gộp chung:", sim.groupby("promo")["churn"].mean().round(3).to_dict())     # promo "có hại"!
print("Theo nhóm rủi ro:\n", sim.groupby(["risk_high", "promo"])["churn"].mean().unstack().round(3))
```

### Inverse Propensity Weighting (IPW)

Ước lượng **propensity score** $e(x) = P(T=1 \mid X=x)$, rồi gán trọng số $1/e(x)$ cho nhóm treatment và $1/(1-e(x))$ cho nhóm control để tạo "quần thể giả" cân bằng về X.

```python
from sklearn.linear_model import LogisticRegression

X_conf = sim[["risk_high"]].astype(int).to_numpy()
e = LogisticRegression().fit(X_conf, sim["promo"]).predict_proba(X_conf)[:, 1]
t, y = sim["promo"].to_numpy(), sim["churn"].to_numpy()
ate_naive = y[t].mean() - y[~t].mean()
ate_ipw = np.mean(t * y / e) - np.mean((1 - t) * y / (1 - e))
print(f"ATE ngây thơ = {ate_naive:+.3f} | ATE IPW = {ate_ipw:+.3f} | ATE thật = -0.050")
```

### Difference-in-Differences (DiD)

Khi một chính sách áp dụng cho một nhóm (vùng A) từ một thời điểm, DiD so sánh **thay đổi** của nhóm được áp dụng với **thay đổi** của nhóm đối chứng. Giả định quan trọng: **xu hướng song song (parallel trends)**.

```python
n_did = 4_000
did = pd.DataFrame({"treated": rng.integers(0, 2, n_did), "post": rng.integers(0, 2, n_did)})
did["churn_rate"] = (0.20 + 0.03 * did["treated"] + 0.02 * did["post"]
                     - 0.04 * did["treated"] * did["post"] + rng.normal(0, 0.05, n_did))
did_fit = smf.ols("churn_rate ~ treated * post", data=did).fit(cov_type="HC3")
print("Hiệu ứng DiD (treated:post):", round(did_fit.params["treated:post"], 4),
      "CI:", np.round(did_fit.conf_int().loc["treated:post"].to_numpy(), 4))
```

| Phương pháp | Giả định chính | Công cụ |
|---|---|---|
| Thí nghiệm ngẫu nhiên (RCT/A/B) | Ngẫu nhiên hóa đúng | statsmodels, scipy |
| Hiệu chỉnh hồi quy / IPW / AIPW | Không có confounder không đo được; **overlap** (0 < e(x) < 1) | DoWhy, EconML |
| Matching (PSM) | Như trên | `causalml`, `DoWhy` |
| DiD | Xu hướng song song | statsmodels, `linearmodels` |
| Regression Discontinuity | Không thao túng được ngưỡng | `rdrobust` |
| Instrumental Variables | Biến công cụ hợp lệ (relevance + exclusion) | `linearmodels.IV2SLS` |
| Synthetic Control | Nhóm đối chứng tổng hợp khớp tiền can thiệp | `pysyncon` |

## 4.9. Phân tích sống còn (Survival analysis): churn theo thời gian

Churn về bản chất là bài toán **thời gian đến sự kiện**. Khách chưa rời bỏ là quan sát **bị kiểm duyệt phải (right-censored)**: ta chỉ biết họ "sống" ít nhất đến hiện tại. Bỏ khách bị kiểm duyệt, hoặc coi họ là "không churn", đều gây bias.

```python
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

n_s = 3_000
m2m = rng.integers(0, 2, n_s)
calls = rng.poisson(1.5, n_s)
true_time = rng.exponential(scale=np.exp(3.5 - 0.8 * m2m - 0.2 * calls))     # tháng đến khi rời bỏ
censor_time = rng.uniform(0, 48, n_s)                                       # thời gian quan sát được
surv = pd.DataFrame({"time": np.minimum(true_time, censor_time),
                     "event": (true_time <= censor_time).astype(int), "m2m": m2m, "calls": calls})
print("Tỷ lệ quan sát bị kiểm duyệt:", round(1 - surv["event"].mean(), 3))

# Kaplan–Meier theo loại hợp đồng + log-rank test
fig, ax = plt.subplots()
for g, sub in surv.groupby("m2m"):
    km = SurvfuncRight(sub["time"], sub["event"])
    km.plot(ax=ax)
    print(f"m2m={g}: median survival = {km.quantile(0.5):.1f} tháng")
chisq, p_lr = survdiff(surv["time"], surv["event"], surv["m2m"])
print(f"Log-rank χ²={chisq:.1f}, p={p_lr:.2e}")

# Cox Proportional Hazards: hazard ratio cho từng biến
cox = PHReg.from_formula("time ~ m2m + calls", data=surv, status=surv["event"].to_numpy()).fit()
print(pd.DataFrame({"HR": np.exp(cox.params), "p": cox.pvalues},
                   index=cox.model.exog_names).round(3))   # HR thật: e^0.8≈2.23, e^0.2≈1.22
```

Thư viện chuyên dụng: **lifelines** (Kaplan–Meier, Cox, AFT, kiểm tra giả định proportional hazards), **scikit-survival** (Random Survival Forest, metric C-index tương thích scikit-learn).

## 4.10. Phân tích chuỗi thời gian cơ bản

```python
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss

days = pd.date_range("2024-01-01", periods=365, freq="D")
signups = pd.Series(200 + 0.2 * np.arange(365) + 25 * np.sin(2 * np.pi * np.arange(365) / 7)
                    + rng.normal(0, 8, 365), index=days)

stl = STL(signups, period=7, robust=True).fit()                  # Trend + Seasonal + Residual
strength_seasonal = max(0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())
print("Độ mạnh mùa vụ tuần (Hyndman):", round(strength_seasonal, 3))
print("ADF p  =", round(adfuller(signups)[1], 4), "(H0: có nghiệm đơn vị / không dừng)")
print("KPSS p =", round(kpss(signups, regression="c", nlags="auto")[1], 4), "(H0: dừng)")
print("ACF lag 7 =", round(acf(signups.diff().dropna(), nlags=7)[7], 3))
```

Kết hợp ADF và KPSS: ADF không bác bỏ **và** KPSS bác bỏ thì chuỗi không dừng, cần sai phân hoặc khử xu hướng trước khi mô hình ARIMA.

> **Checklist Chương 4**
> - [ ] Phát biểu giả thuyết và α **trước khi** nhìn dữ liệu; chọn kiểm định phù hợp kiểu dữ liệu và giả định.
> - [ ] Báo cáo **effect size + CI**, không chỉ p-value; diễn giải p-value đúng tinh thần ASA.
> - [ ] Hiệu chỉnh đa kiểm định (Holm/BH) khi kiểm định nhiều giả thuyết.
> - [ ] A/B test: tính cỡ mẫu trước, kiểm tra SRM, không nhìn trộm, dùng CUPED khi có covariate.
> - [ ] Hồi quy: kiểm tra giả định, dùng robust SE khi cần, diễn giải bằng OR / marginal effects.
> - [ ] Không diễn giải nhân quả từ dữ liệu quan sát khi chưa xử lý confounder.
> - [ ] Dữ liệu thời gian-đến-sự-kiện được xử lý bằng phương pháp survival (tôn trọng kiểm duyệt).

### Tài liệu tham khảo Chương 4

- Wasserstein, R. L. & Lazar, N. A. (2016). *The ASA Statement on p-Values: Context, Process, and Purpose.* The American Statistician 70(2).
- SciPy reference: `scipy.stats` — *Hypothesis tests*, `bootstrap`, `permutation_test`. docs.scipy.org
- statsmodels User Guide: *Regression*, *GLM*, *Stats (power, proportion, multitest)*, *Duration (survival)*, *Time Series Analysis*. statsmodels.org
- Kohavi, R., Tang, D. & Xu, Y. (2020). *Trustworthy Online Controlled Experiments.* Cambridge University Press.
- Deng, A. et al. (2013). *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data (CUPED).* WSDM.
- Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.
- Benjamini, Y. & Hochberg, Y. (1995). *Controlling the False Discovery Rate.* JRSS-B 57(1).
- Hernán, M. A. & Robins, J. M. (2020). *Causal Inference: What If.* Chapman & Hall/CRC (bản miễn phí).
- Cunningham, S. (2021). *Causal Inference: The Mixtape.* Yale University Press.
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. (otexts.com/fpp3)
- Harvard CS109, Stanford STATS 200/STATS 361 (tài liệu bài giảng về suy luận và nhân quả).
