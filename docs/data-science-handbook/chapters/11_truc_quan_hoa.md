# CHƯƠNG 11. TRỰC QUAN HÓA DỮ LIỆU & KẾT QUẢ

## 11.1. Nguyên tắc thiết kế biểu đồ

1. **Một biểu đồ — một thông điệp.** Tiêu đề nên là kết luận ("Khách hợp đồng tháng churn gấp 3 lần"), không phải mô tả ("Churn theo hợp đồng").
2. **Tối đa hóa tỷ lệ data-ink** (Tufte): bỏ viền, lưới nặng, hiệu ứng 3D, màu thừa.
3. **Trục bắt đầu từ 0 với biểu đồ cột.** Biểu đồ đường có thể không.
4. **Màu có chủ đích:** xám cho ngữ cảnh, một màu nhấn cho điểm cần chú ý; bảng màu thân thiện người mù màu (`viridis`, `cividis`, palette `colorblind`).
5. **Ghi chú trực tiếp** (direct labeling) thay vì bắt người đọc dò chú thích.
6. Luôn ghi **đơn vị, nguồn dữ liệu, thời gian, cỡ mẫu**.

## 11.2. Chọn biểu đồ theo mục đích

| Mục đích | Biểu đồ phù hợp | Tránh |
|---|---|---|
| Phân phối 1 biến số | Histogram, KDE, box/violin, ECDF | Pie chart |
| So sánh phân phối giữa nhóm | Box, violin, ridgeline, ECDF chồng | Nhiều histogram chồng đặc |
| So sánh giá trị giữa các nhóm | Bar (ngang nếu nhãn dài), dot plot | Pie > 5 phần, 3D bar |
| Thành phần (tỷ lệ) | Stacked bar 100%, treemap | Pie nhiều lát |
| Xu hướng theo thời gian | Line, area | Bar cho chuỗi dài |
| Quan hệ 2 biến số | Scatter (+ hexbin khi nhiều điểm), regression line | |
| Tương quan nhiều biến | Heatmap, pairplot | |
| Dữ liệu địa lý | Choropleth, bubble map | |
| Đánh giá mô hình | ROC, PR, calibration, confusion matrix, lift, residual | |

## 11.3. Thiết lập style chung cho toàn dự án

```python
# src/churn/viz/style.py
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = {"primary": "#1f5fa8", "accent": "#d1495b", "neutral": "#9aa5b1", "good": "#2a9d8f"}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    mpl.rcParams.update({
        "figure.figsize": (10, 5.5), "figure.dpi": 110, "savefig.dpi": 200,
        "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
        "font.family": "DejaVu Sans",   # hỗ trợ tiếng Việt có dấu
    })


def save(fig, name: str) -> None:
    fig.savefig(f"reports/figures/{name}.png")
    fig.savefig(f"reports/figures/{name}.svg")
```

## 11.4. Biểu đồ EDA (matplotlib + seaborn)

```python
import numpy as np
import pandas as pd

from churn.data.synthetic import make_churn_data
from churn.features.clean import clean_churn
from churn.viz.style import PALETTE, mpl, plt, save, set_style, sns

set_style()
df = clean_churn(make_churn_data())

# 1) Phân phối + so sánh theo target: histogram chồng mật độ
fig, ax = plt.subplots()
sns.histplot(data=df, x="monthly_charges", hue="churn", stat="density", common_norm=False,
             element="step", bins=50, ax=ax)
ax.set(title="Khách churn có cước tháng cao hơn", xlabel="Cước tháng (nghìn VNĐ)",
       ylabel="Mật độ")

# 2) ECDF — so sánh phân phối chính xác hơn histogram (không phụ thuộc số bin)
fig, ax = plt.subplots()
sns.ecdfplot(data=df, x="tenure_months", hue="churn", ax=ax)
ax.set(title="50% khách churn có thời gian gắn bó < 1 năm")

# 3) Tỷ lệ churn theo nhóm, có khoảng tin cậy (seaborn tự bootstrap CI)
fig, ax = plt.subplots()
order = df.groupby("contract")["churn"].mean().sort_values(ascending=False).index
sns.barplot(data=df, x="contract", y="churn", order=order, errorbar=("ci", 95),
            color=PALETTE["primary"], ax=ax)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
for c in ax.containers:
    ax.bar_label(c, fmt=lambda v: f"{v:.1%}", padding=3)
ax.set(title="Hợp đồng theo tháng có tỷ lệ churn cao nhất", xlabel="", ylabel="Tỷ lệ churn")

# 4) Violin theo nhóm
fig, ax = plt.subplots()
sns.violinplot(data=df, x="contract", y="support_calls", hue="churn", split=True,
               inner="quart", ax=ax)

# 5) Heatmap tương quan (tam giác dưới)
num = df.select_dtypes("number").drop(columns=["churn"])
corr = num.corr(method="spearman")
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, bool)), annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
ax.set_title("Tương quan Spearman")

# 6) Pairplot (lấy mẫu để nhanh)
sns.pairplot(df.sample(1500, random_state=0),
             vars=["tenure_months", "monthly_charges", "support_calls"],
             hue="churn", corner=True, plot_kws={"s": 8, "alpha": 0.5})

# 7) Chuỗi thời gian với highlight
monthly = df.set_index("signup_date").resample("MS")["churn"].mean()
fig, ax = plt.subplots()
ax.plot(monthly.index, monthly.values, color=PALETTE["neutral"])
peak = monthly.idxmax()
ax.scatter([peak], [monthly.max()], color=PALETTE["accent"], zorder=3)
ax.annotate(f"Đỉnh {monthly.max():.1%}", (peak, monthly.max()), xytext=(10, 10),
            textcoords="offset points")
ax.set(title="Tỷ lệ churn theo cohort tháng đăng ký")
```

## 11.5. Dashboard biểu đồ đánh giá mô hình (một hình, 6 panel)

```python
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             precision_recall_curve, roc_auc_score, roc_curve)


def model_dashboard(y_true, proba, threshold: float = 0.5, title: str = "Model evaluation"):
    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # (1) ROC
    fpr, tpr, _ = roc_curve(y_true, proba)
    ax = axes[0, 0]
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(title="ROC curve", xlabel="FPR", ylabel="TPR"); ax.legend(loc="lower right")

    # (2) Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ax = axes[0, 1]
    ax.plot(rec, prec, label=f"AP = {average_precision_score(y_true, proba):.3f}")
    ax.axhline(y_true.mean(), ls="--", color="grey", label=f"Baseline = {y_true.mean():.3f}")
    ax.set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision"); ax.legend()

    # (3) Confusion matrix tại ngưỡng
    ConfusionMatrixDisplay.from_predictions(y_true, (proba >= threshold).astype(int),
                                            display_labels=["stay", "churn"], cmap="Blues",
                                            colorbar=False, ax=axes[0, 2])
    axes[0, 2].set_title(f"Confusion matrix @ {threshold:.2f}")

    # (4) Phân phối điểm theo lớp
    ax = axes[1, 0]
    ax.hist(proba[y_true == 0], bins=50, alpha=0.6, density=True, label="stay")
    ax.hist(proba[y_true == 1], bins=50, alpha=0.6, density=True, label="churn")
    ax.axvline(threshold, color="k", ls="--")
    ax.set(title="Phân phối điểm dự đoán", xlabel="P(churn)"); ax.legend()

    # (5) Calibration
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    ax = axes[1, 1]
    ax.plot(mean_pred, frac_pos, "o-"); ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(title="Calibration (reliability)", xlabel="Xác suất dự đoán", ylabel="Tỷ lệ thực tế")

    # (6) Cumulative gain
    order = np.argsort(-proba)
    gains = np.cumsum(y_true[order]) / y_true.sum()
    pct = np.arange(1, len(y_true) + 1) / len(y_true)
    ax = axes[1, 2]
    ax.plot(pct, gains, label="Model"); ax.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    ax.set(title="Cumulative gain", xlabel="% khách được liên hệ", ylabel="% churn bắt được")
    ax.legend()

    fig.tight_layout()
    return fig


# fig = model_dashboard(y_valid, proba_valid, threshold=0.35); save(fig, "model_dashboard")
```

## 11.6. Biểu đồ tương tác với Plotly

```python
import plotly.express as px
import plotly.graph_objects as go

s = df.sample(3000, random_state=0).assign(label=lambda d: d["churn"].map({0: "stay", 1: "churn"}))
fig = px.scatter(s, x="tenure_months", y="monthly_charges", color="label",
                 hover_data=["customer_id", "contract"], opacity=0.6,
                 title="Tenure vs Cước tháng", template="plotly_white")
fig.write_html("reports/figures/scatter.html", include_plotlyjs="cdn")

# Funnel / sunburst cho phân khúc
seg = df.groupby(["contract", "payment_method"], observed=True).agg(n=("churn", "size"),
                                                                    churn_rate=("churn", "mean")).reset_index()
fig = px.sunburst(seg, path=["contract", "payment_method"], values="n", color="churn_rate",
                  color_continuous_scale="RdYlGn_r")
```

## 11.7. Dashboard ứng dụng với Streamlit

```python
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
def load_data() -> pd.DataFrame:
    return pd.read_parquet("data/processed/scoring_latest.parquet")


model, data = load_model(), load_data()
data["score"] = model.predict_proba(data)[:, 1]

st.title("📉 Customer Churn Monitor")
threshold = st.sidebar.slider("Ngưỡng rủi ro", 0.05, 0.95, 0.35, 0.05)
contracts = st.sidebar.multiselect("Loại hợp đồng", sorted(data["contract"].dropna().unique()))
view = data[data["contract"].isin(contracts)] if contracts else data

c1, c2, c3 = st.columns(3)
c1.metric("Số khách", f"{len(view):,}")
c2.metric("Nguy cơ cao", f"{(view['score'] >= threshold).sum():,}")
c3.metric("Điểm trung bình", f"{view['score'].mean():.1%}")

st.plotly_chart(px.histogram(view, x="score", nbins=50, title="Phân phối điểm rủi ro"),
                use_container_width=True)
st.dataframe(view.nlargest(100, "score")[["customer_id", "contract", "tenure_months", "score"]])
st.download_button("Tải danh sách gọi điện (CSV)",
                   view[view["score"] >= threshold].to_csv(index=False).encode("utf-8"),
                   file_name="call_list.csv")
```

Công cụ BI cho người dùng nghiệp vụ: **Power BI, Tableau, Looker, Apache Superset, Metabase**.

## 11.8. Kể chuyện bằng dữ liệu (Data Storytelling)

Cấu trúc trình bày kết quả cho lãnh đạo:

1. **Bối cảnh** — vấn đề kinh doanh và vì sao quan trọng (con số tiền).
2. **Phát hiện chính** — 3 insight, mỗi insight 1 biểu đồ có tiêu đề là kết luận.
3. **Giải pháp** — mô hình làm gì, hiệu quả so với cách hiện tại (lift, lợi nhuận kỳ vọng).
4. **Hành động đề xuất** — ai làm gì, khi nào, đo lường thế nào.
5. **Rủi ro & bước tiếp theo.**

> **Checklist Chương 11**
> - [ ] Mỗi biểu đồ có tiêu đề là thông điệp, có đơn vị và nguồn.
> - [ ] Loại biểu đồ phù hợp mục đích; tránh pie nhiều lát, 3D.
> - [ ] Bảng màu thống nhất, thân thiện người mù màu; font hỗ trợ tiếng Việt.
> - [ ] Có dashboard đánh giá mô hình chuẩn (ROC, PR, CM, calibration, gain).
> - [ ] Biểu đồ được lưu tự động (PNG + SVG) từ code, không chụp màn hình thủ công.
