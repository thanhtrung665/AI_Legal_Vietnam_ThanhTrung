# CHƯƠNG 11. TRỰC QUAN HÓA DỮ LIỆU & KỂ CHUYỆN BẰNG DỮ LIỆU

**Mục tiêu chương:** chọn đúng loại biểu đồ cho đúng câu hỏi, dựng biểu đồ **chính xác về nhận thức**, nhất quán về phong cách, tái lập được bằng code, và trình bày kết quả để **dẫn tới quyết định**.

## 11.1. Nền tảng nhận thức thị giác

**Thứ bậc độ chính xác khi giải mã (Cleveland & McGill, 1984)**, từ chính xác nhất đến kém nhất:

1. Vị trí trên cùng một trục (dot plot, scatter, bar)
2. Vị trí trên các trục không thẳng hàng (small multiples)
3. Độ dài (bar)
4. Góc, độ dốc
5. Diện tích (bubble, treemap)
6. Thể tích, độ cong
7. Độ đậm nhạt, độ bão hòa màu

**Hệ quả:** pie chart (góc/diện tích) kém hơn bar chart (độ dài/vị trí) khi so sánh giá trị. Heatmap (màu) phù hợp cho *mẫu hình*, không phù hợp để đọc giá trị chính xác.

**Nguyên tắc của Tufte** (*The Visual Display of Quantitative Information*, 1983):

- **Data-ink ratio:** tối đa hóa phần mực dùng để thể hiện dữ liệu; bỏ khung, lưới đậm, hiệu ứng 3D, bóng đổ.
- **Không bóp méo:** "lie factor" = độ lớn hiệu ứng trên hình / độ lớn trong dữ liệu ≈ 1. Biểu đồ cột **phải bắt đầu từ 0**.
- **Small multiples:** nhiều biểu đồ nhỏ cùng thang đo thường tốt hơn một biểu đồ chồng chất.

**Grammar of Graphics (Wilkinson, 2005):** một biểu đồ = **dữ liệu** + **ánh xạ thẩm mỹ** (x, y, màu, kích thước) + **hình học** (điểm, đường, cột) + **thống kê** (bin, smooth) + **hệ tọa độ** + **facet**. Đây là nền tảng của ggplot2, `seaborn.objects`, plotnine, Altair/Vega-Lite.

## 11.2. Chọn biểu đồ theo câu hỏi

| Câu hỏi | Biểu đồ phù hợp | Tránh |
|---|---|---|
| Phân phối một biến số | Histogram (bin FD), KDE, **ECDF**, box/violin | Pie |
| So sánh phân phối giữa nhóm | Box/violin cạnh nhau, ECDF chồng, ridgeline | Nhiều histogram chồng đặc |
| So sánh giá trị giữa nhóm | **Bar ngang** (nhãn dài), **dot plot**, có khoảng tin cậy | 3D bar, pie > 4 phần |
| Thành phần (tỷ lệ) | Stacked bar 100%, waffle, treemap | Pie nhiều lát |
| Xu hướng theo thời gian | Line (+ dải CI), area | Bar cho chuỗi dài |
| Quan hệ 2 biến số | Scatter (+ hexbin/2D KDE khi nhiều điểm), đường hồi quy/LOWESS | Nối điểm không có thứ tự |
| Nhiều biến | Heatmap tương quan, pairplot, parallel coordinates | |
| Địa lý | Choropleth (chuẩn hóa theo dân số!), bubble map | Choropleth số tuyệt đối |
| Đánh giá mô hình | ROC, PR, calibration, confusion matrix, lift/gain, residual | |
| Bất định | Error bar, dải CI, fan chart, gradient interval | Chỉ vẽ ước lượng điểm |

## 11.3. Kiến trúc matplotlib và phong cách thống nhất

**[Docs]** matplotlib phân biệt hai giao diện. **Giao diện hướng đối tượng (OO)** (`fig, ax = plt.subplots()`, gọi phương thức trên `ax`) được **khuyến nghị** cho code tái sử dụng và biểu đồ phức tạp. **Giao diện pyplot** (state-based, `plt.plot`) chỉ phù hợp khi vẽ nhanh tương tác. Cấu trúc đối tượng: **Figure** → **Axes** (một hệ trục) → **Axis**, **Artist** (mọi thứ được vẽ).

Package tham chiếu có `churn/viz/style.py` để mọi biểu đồ trong dự án dùng chung phong cách:

```python norun
PALETTE = {"primary": "#1f5fa8", "accent": "#d1495b", "neutral": "#9aa5b1", "good": "#2a9d8f"}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    mpl.rcParams.update({
        "figure.figsize": (10, 5.5), "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "font.family": "DejaVu Sans",      # font hỗ trợ tiếng Việt
    })


def save(fig, name, folder="reports/figures"):        # PNG cho slide + SVG cho in ấn
    for ext in ("png", "svg"):
        fig.savefig(Path(folder) / f"{name}.{ext}")
```

### Thiết lập

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from churn.data.synthetic import make_churn_data
from churn.features.clean import clean_churn
from churn.viz.style import PALETTE, save, set_style

set_style()
df = clean_churn(make_churn_data(n=20_000, seed=42))
df["churn_label"] = df["churn"].map({0: "Ở lại", 1: "Rời bỏ"})
```

## 11.4. Màu sắc có chủ đích

| Loại bảng màu | Dùng cho | Ví dụ (matplotlib/seaborn) |
|---|---|---|
| **Qualitative** | Nhóm không có thứ tự | `colorblind`, `tab10`, `Set2` |
| **Sequential** | Giá trị từ thấp đến cao | `viridis`, `cividis`, `Blues` |
| **Diverging** | Có điểm giữa ý nghĩa (0, trung bình) | `RdBu_r`, `coolwarm` (đặt `center=0`) |

Nguyên tắc:

- Dùng **màu xám cho ngữ cảnh** và **một màu nhấn** cho điều cần chú ý.
- Bảng màu **perceptually uniform** (`viridis`, `cividis`) giữ thứ tự khi in đen trắng và với người mù màu (khoảng 8% nam giới).
- Tránh `jet`/`rainbow`: tạo ranh giới giả và không đơn điệu về độ sáng.
- Không mã hóa cùng một biến bằng hai màu khác nhau ở hai biểu đồ cạnh nhau.

```python
fig, ax = plt.subplots(figsize=(9, 4.5))
rates = df.groupby("region")["churn"].mean().sort_values()
colors = [PALETTE["accent"] if r == rates.idxmax() else PALETTE["neutral"] for r in rates.index]
ax.barh(rates.index, rates.values, color=colors)
ax.axvline(df["churn"].mean(), color="k", lw=1, ls="--")
ax.text(df["churn"].mean(), len(rates) - 0.5, " trung bình", va="top", fontsize=9)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set(title=f"Vùng {rates.idxmax()} cao nhất, nhưng mọi vùng nằm trong dao động ngẫu nhiên",
       xlabel="Tỷ lệ churn", ylabel="")
ax.tick_params(axis="y", labelsize=7)
save(fig, "churn_by_region")
```

> Tiêu đề ở đây nói đúng **sự thật thống kê**: `region` không có tác động thật (Chương 0.8). Biểu đồ đặt màu nhấn vào vùng cao nhất **rất dễ gây hiểu lầm** nếu không kèm khoảng tin cậy. Xem cách sửa ở mục 11.5.

## 11.5. Biểu đồ EDA chuẩn mực

### So sánh nhóm phải kèm độ bất định

```python
from statsmodels.stats.proportion import proportion_confint

g = df.groupby("region")["churn"].agg(["sum", "count"])
lo, hi = proportion_confint(g["sum"], g["count"], method="wilson")
g = g.assign(rate=g["sum"] / g["count"], lo=lo, hi=hi).sort_values("rate")
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(g["rate"], range(len(g)), xerr=[g["rate"] - g["lo"], g["hi"] - g["rate"]], fmt="o",
            color=PALETTE["primary"], ecolor=PALETTE["neutral"], capsize=2, ms=4)
ax.axvline(df["churn"].mean(), color=PALETTE["accent"], ls="--", lw=1)
ax.set_yticks(range(len(g)), g.index, fontsize=7)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set(title="Khoảng tin cậy 95% của các vùng chồng lên nhau: không có vùng nào khác biệt thật",
       xlabel="Tỷ lệ churn (Wilson 95% CI)")
```

### Phân phối theo nhóm: histogram mật độ, ECDF, violin

```python
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
sns.histplot(data=df, x="monthly_charges", hue="churn_label", stat="density", common_norm=False,
             element="step", bins="fd", ax=axes[0])
axes[0].set_xlim(0, 200)
axes[0].set(title="Khách rời bỏ có cước cao hơn", xlabel="Cước tháng (nghìn VNĐ)")
sns.ecdfplot(data=df, x="tenure_months", hue="churn_label", ax=axes[1])
axes[1].axhline(0.5, color="grey", lw=0.8, ls=":")
axes[1].set(title="ECDF: median tenure của nhóm rời bỏ thấp hơn rõ", xlabel="Tenure (tháng)")
sns.violinplot(data=df, x="contract", y="support_calls", hue="churn_label", split=True, inner="quart",
               order=["month-to-month", "one-year", "two-year"], ax=axes[2])
axes[2].set(title="Số cuộc gọi hỗ trợ theo hợp đồng", xlabel="")
plt.tight_layout()
save(fig, "eda_distributions")
```

**[Docs]** seaborn phân biệt hàm **axes-level** (`histplot`, `scatterplot`… vẽ lên một `ax`) và **figure-level** (`displot`, `relplot`, `catplot`… tự tạo figure, hỗ trợ facet qua `col`/`row`). Dùng figure-level cho **small multiples**:

```python
grid = sns.displot(data=df, x="tenure_months", hue="churn_label", col="contract", kind="ecdf", height=3.6,
                   aspect=1.1, col_order=["month-to-month", "one-year", "two-year"])
grid.set_titles("{col_name}")
grid.figure.suptitle("Small multiples: tenure theo từng loại hợp đồng", y=1.05)
```

### Tỷ lệ theo biến liên tục (binned), có CI

```python
fig, ax = plt.subplots(figsize=(8, 4.5))
for contract, sub in df.groupby("contract"):
    b = sub.assign(bin=pd.qcut(sub["tenure_months"], 8, duplicates="drop")).groupby("bin", observed=True)["churn"]
    stats_ = b.agg(["sum", "count"])
    lo, hi = proportion_confint(stats_["sum"], stats_["count"], method="wilson")
    mid = [iv.mid for iv in stats_.index]
    ax.plot(mid, stats_["sum"] / stats_["count"], marker="o", label=contract)
    ax.fill_between(mid, lo, hi, alpha=0.15)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set(title="Churn giảm theo tenure ở mọi loại hợp đồng", xlabel="Tenure (tháng)", ylabel="Tỷ lệ churn")
ax.legend(title="Hợp đồng", frameon=False)
```

### Quan hệ nhiều biến

```python
num = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls", "data_usage_gb"]
corr = df[num].corr(method="spearman")
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Tương quan Spearman (tam giác dưới)")

sample = df.sample(1500, random_state=0)
pg = sns.pairplot(sample, vars=["tenure_months", "monthly_charges", "support_calls"], hue="churn_label",
                  corner=True, plot_kws={"s": 8, "alpha": 0.4}, diag_kind="kde")
pg.figure.suptitle("Pairplot (mẫu 1 500)", y=1.02)
```

### Chuỗi thời gian có chú thích

```python
monthly = df.set_index("signup_date").resample("MS")["churn"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(monthly.index, monthly["mean"], color=PALETTE["neutral"], marker="o", ms=3)
roll = monthly["mean"].rolling(3, center=True).mean()
ax.plot(monthly.index, roll, color=PALETTE["primary"], lw=2.5, label="Trung bình trượt 3 tháng")
peak = monthly["mean"].idxmax()
ax.annotate(f"Đỉnh {monthly['mean'].max():.1%}", xy=(peak, monthly["mean"].max()), xytext=(15, 10),
            textcoords="offset points", arrowprops={"arrowstyle": "->", "color": PALETTE["accent"]})
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set(title="Tỷ lệ churn theo cohort tháng đăng ký", ylabel="Tỷ lệ churn")
ax.legend(frameon=False)
```

## 11.6. Dashboard đánh giá mô hình (một hình, sáu panel)

**[Docs]** scikit-learn cung cấp **Display API** thống nhất cho biểu đồ đánh giá: `RocCurveDisplay`, `PrecisionRecallDisplay`, `DetCurveDisplay`, `ConfusionMatrixDisplay`, `CalibrationDisplay`, `PredictionErrorDisplay`, `LearningCurveDisplay`, `ValidationCurveDisplay`, `PartialDependenceDisplay`. Mỗi lớp có `from_estimator(...)` (tự tính dự đoán) và `from_predictions(...)` (dùng dự đoán có sẵn, nên dùng khi dự đoán tốn kém).

```python
import lightgbm as lgb
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.pipeline import make_pipeline

from churn.data.split import time_split
from churn.features.build import RAW_FEATURES, build_preprocessor

tr, va, te = time_split(df, "signup_date", "2023-10-01", "2024-01-01")
model = make_pipeline(build_preprocessor(scale=False), lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, verbose=-1, random_state=0)
).fit(tr[RAW_FEATURES], tr["churn"])
y_true, proba = te["churn"].to_numpy(), model.predict_proba(te[RAW_FEATURES])[:, 1]


def model_dashboard(y_true: np.ndarray, proba: np.ndarray, threshold: float, title: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    RocCurveDisplay.from_predictions(y_true, proba, ax=axes[0, 0], plot_chance_level=True)
    axes[0, 0].set_title("ROC")
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=axes[0, 1], plot_chance_level=True)
    axes[0, 1].set_title("Precision–Recall")
    ConfusionMatrixDisplay.from_predictions(y_true, (proba >= threshold).astype(int), display_labels=["Ở lại", "Rời bỏ"],
                                            cmap="Blues", colorbar=False, ax=axes[0, 2])
    axes[0, 2].set_title(f"Confusion matrix @ {threshold:.2f}")
    for label, name in [(0, "Ở lại"), (1, "Rời bỏ")]:
        axes[1, 0].hist(proba[y_true == label], bins=40, density=True, alpha=0.55, label=name)
    axes[1, 0].axvline(threshold, color="k", ls="--")
    axes[1, 0].set(title="Phân phối điểm theo lớp", xlabel="P(churn)")
    axes[1, 0].legend(frameon=False)
    CalibrationDisplay.from_predictions(y_true, proba, n_bins=10, strategy="quantile", ax=axes[1, 1])
    axes[1, 1].set_title("Calibration (reliability)")
    order = np.argsort(-proba)
    gain = np.cumsum(y_true[order]) / y_true.sum()
    frac = np.arange(1, len(y_true) + 1) / len(y_true)
    axes[1, 2].plot(frac, gain, label="Mô hình")
    axes[1, 2].plot([0, 1], [0, 1], "k--", label="Ngẫu nhiên")
    axes[1, 2].set(title="Cumulative gain", xlabel="% khách được liên hệ", ylabel="% churn bắt được")
    axes[1, 2].legend(frameon=False)
    fig.tight_layout()
    return fig


fig = model_dashboard(y_true, proba, threshold=0.33, title="Churn model — test out-of-time (2024-H1)")
save(fig, "model_dashboard")
```

### Biểu đồ phần dư cho hồi quy

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import PredictionErrorDisplay

reg = HistGradientBoostingRegressor(random_state=0).fit(tr[["tenure_months", "monthly_charges"]], tr["total_charges"])
pred = reg.predict(te[["tenure_months", "monthly_charges"]])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
PredictionErrorDisplay.from_predictions(te["total_charges"], pred, kind="actual_vs_predicted", subsample=1000,
                                        ax=axes[0], random_state=0)
PredictionErrorDisplay.from_predictions(te["total_charges"], pred, kind="residual_vs_predicted", subsample=1000,
                                        ax=axes[1], random_state=0)
axes[0].set_title("Thực tế vs dự đoán")
axes[1].set_title("Phần dư vs dự đoán (hình phễu = phương sai không đều)")
plt.tight_layout()
```

## 11.7. Biểu đồ tương tác với Plotly

```python
import plotly.express as px

s = df.sample(3000, random_state=0)
fig_px = px.scatter(s, x="tenure_months", y="monthly_charges", color="churn_label", opacity=0.6,
                    hover_data=["customer_id", "contract", "support_calls"], template="simple_white",
                    color_discrete_map={"Ở lại": PALETTE["neutral"], "Rời bỏ": PALETTE["accent"]},
                    title="Tenure vs cước tháng (di chuột để xem chi tiết)")
fig_px.update_yaxes(range=[0, 200])
fig_px.write_html("reports/figures/scatter_interactive.html", include_plotlyjs="cdn")

seg = (df.groupby(["contract", "payment_method"], observed=True)
         .agg(n=("churn", "size"), churn_rate=("churn", "mean")).reset_index())
fig_sb = px.sunburst(seg, path=["contract", "payment_method"], values="n", color="churn_rate",
                     color_continuous_scale="RdYlGn_r", title="Phân khúc: kích thước = số khách, màu = churn")
print(len(fig_sb.data[0]["ids"]), "nút trong sunburst")
```

## 11.8. Dashboard ứng dụng với Streamlit

**[Docs]** Streamlit chạy lại toàn bộ script mỗi khi người dùng tương tác. Vì vậy phải **cache** đúng cách: `st.cache_resource` cho đối tượng dùng chung không nên copy (mô hình, kết nối DB); `st.cache_data` cho dữ liệu trả về có thể serialize (DataFrame), có `ttl`.

```python norun
# app/dashboard.py  —  chạy: streamlit run app/dashboard.py
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Churn Monitor", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")


@st.cache_data(ttl=3600)
def load_scores() -> pd.DataFrame:
    return pd.read_parquet("data/processed/scoring_latest.parquet")


model, data = load_model(), load_scores()
data["score"] = model.predict_proba(data)[:, 1]

st.title("Customer Churn Monitor")
threshold = st.sidebar.slider("Ngưỡng rủi ro", 0.05, 0.95, 0.33, 0.01)
contracts = st.sidebar.multiselect("Loại hợp đồng", sorted(data["contract"].dropna().unique()))
view = data[data["contract"].isin(contracts)] if contracts else data

c1, c2, c3 = st.columns(3)
c1.metric("Số khách", f"{len(view):,}")
c2.metric("Nguy cơ cao", f"{(view['score'] >= threshold).sum():,}")
c3.metric("Điểm trung bình", f"{view['score'].mean():.1%}")
st.plotly_chart(px.histogram(view, x="score", nbins=50), use_container_width=True)
st.dataframe(view.nlargest(100, "score")[["customer_id", "contract", "tenure_months", "score"]])
st.download_button("Tải danh sách gọi điện (CSV)",
                   view[view["score"] >= threshold].to_csv(index=False).encode("utf-8"), "call_list.csv")
```

Công cụ BI cho người dùng nghiệp vụ: **Power BI, Tableau, Looker, Apache Superset, Metabase**. Data Scientist nên cung cấp **bảng điểm đã tính sẵn** (batch scoring, Chương 12) cho các công cụ này, thay vì để BI gọi mô hình.

## 11.9. Kể chuyện bằng dữ liệu (Data Storytelling)

**[Sách]** Cole Nussbaumer Knaflic, *Storytelling with Data* (2015), sáu bài học:

1. **Hiểu bối cảnh:** ai là khán giả, họ cần quyết định gì, bạn muốn họ làm gì.
2. **Chọn hình ảnh phù hợp** (mục 11.2).
3. **Loại bỏ rối mắt (clutter).**
4. **Hướng sự chú ý:** màu nhấn, kích thước, vị trí.
5. **Tư duy như nhà thiết kế:** căn lề, khoảng trắng, phân cấp chữ.
6. **Kể một câu chuyện:** mở đầu (bối cảnh), cao trào (insight), kết thúc (hành động).

### Cấu trúc bài trình bày cho lãnh đạo (một trang / năm slide)

| Phần | Nội dung | Ví dụ |
|---|---|---|
| 1. Vấn đề | Bối cảnh + con số tiền | "Mỗi tháng mất 2.1% thuê bao ≈ 12 tỷ doanh thu/năm" |
| 2. Phát hiện | 3 insight, mỗi insight 1 biểu đồ có tiêu đề là kết luận | "Khách hợp đồng tháng rời bỏ gấp 3 lần" |
| 3. Giải pháp | Mô hình làm gì, so với cách hiện tại | "Gọi top 10% bắt được 30% khách sắp rời bỏ, gấp 3 lần quy tắc hiện tại" |
| 4. Đề xuất | Ai làm gì, khi nào, đo thế nào | "A/B test 4 tuần, 2 × 5 500 khách, đo tỷ lệ giữ chân 60 ngày" |
| 5. Rủi ro & bước tiếp | Hạn chế, kế hoạch giám sát | "Kém chính xác với khách < 3 tháng; giám sát PSI hằng tuần" |

**Tiêu đề biểu đồ là câu kết luận** ("Khách hợp đồng tháng rời bỏ gấp 3 lần"), không phải mô tả ("Churn theo hợp đồng").

> **Checklist Chương 11**
> - [ ] Loại biểu đồ phù hợp câu hỏi và thứ bậc nhận thức; không dùng pie nhiều lát, 3D, trục cột không bắt đầu từ 0.
> - [ ] So sánh nhóm luôn kèm độ bất định (CI); không đặt màu nhấn vào khác biệt do ngẫu nhiên.
> - [ ] Bảng màu đúng loại (qualitative/sequential/diverging), thân thiện người mù màu; font hỗ trợ tiếng Việt.
> - [ ] Dùng giao diện OO của matplotlib; phong cách chung qua một module; biểu đồ lưu tự động (PNG + SVG).
> - [ ] Dashboard đánh giá mô hình chuẩn (ROC, PR, CM, phân phối điểm, calibration, gain) bằng Display API.
> - [ ] Tiêu đề là kết luận; có đơn vị, nguồn, giai đoạn, cỡ mẫu.
> - [ ] Bài trình bày đi từ vấn đề → insight → giải pháp → hành động → rủi ro.

### Tài liệu tham khảo Chương 11

- matplotlib documentation: *Quick start guide* (Figure/Axes anatomy, OO vs pyplot), *Choosing Colormaps*. matplotlib.org
- seaborn documentation: *Overview of seaborn plotting functions* (axes-level vs figure-level), *Visualizing distributions*, *Choosing color palettes*, *The seaborn.objects interface*. seaborn.pydata.org
- scikit-learn User Guide: *Visualizations* (Display objects). · Plotly Python documentation. · Streamlit documentation: *Caching*.
- Cleveland, W. S. & McGill, R. (1984). *Graphical Perception.* JASA 79(387).
- Tufte, E. R. (2001). *The Visual Display of Quantitative Information*, 2nd ed. Graphics Press.
- Wilkinson, L. (2005). *The Grammar of Graphics*, 2nd ed. Springer.
- Wilke, C. O. (2019). *Fundamentals of Data Visualization.* O'Reilly (clauswilke.com/dataviz).
- Knaflic, C. N. (2015). *Storytelling with Data.* Wiley.
- Harvard CS109: *Visualization* lectures.
