# CHƯƠNG 10. GIẢI THÍCH MÔ HÌNH (XAI) & FAIRNESS

**Mục tiêu chương:** trả lời **vì sao** mô hình dự đoán như vậy, ở cấp toàn cục lẫn từng khách hàng; phát hiện mô hình học "sai lý do" (leakage, proxy); và kiểm tra mô hình có **công bằng** giữa các nhóm không.

## 10.1. Vì sao cần giải thích và giải thích cho ai?

| Người dùng | Câu hỏi | Kỹ thuật phù hợp |
|---|---|---|
| Data Scientist | Mô hình có học đúng quy luật không? Có leakage không? | Permutation importance, PDP/ICE, SHAP, error analysis |
| Nhân viên vận hành (CSKH) | Vì sao khách này rủi ro cao? Nên nói gì với khách? | Reason codes từ SHAP; counterfactual |
| Lãnh đạo / nghiệp vụ | Yếu tố nào thúc đẩy churn? | Global importance + PDP, có diễn giải nghiệp vụ |
| Kiểm toán / pháp chế | Quyết định có giải thích được, có phân biệt đối xử không? | Model card, fairness metrics, mô hình glass-box |

**Phân loại phương pháp** (Molnar, *Interpretable Machine Learning*):

| Trục | Lựa chọn |
|---|---|
| Bản chất | **Intrinsic** (mô hình tự giải thích được: tuyến tính, cây nông, GAM/EBM) và **post-hoc** (giải thích mô hình hộp đen) |
| Phạm vi | **Global** (toàn mô hình) và **local** (một dự đoán) |
| Phụ thuộc mô hình | **Model-specific** (TreeSHAP, hệ số) và **model-agnostic** (permutation, PDP, KernelSHAP, LIME) |

> **Cảnh báo quan trọng.** Mọi phương pháp ở chương này giải thích **mô hình**, không giải thích **thế giới thật**. Feature quan trọng với mô hình **không có nghĩa** là thay đổi feature đó sẽ thay đổi kết quả thật. Muốn kết luận nhân quả, xem Chương 4.8.

### Thiết lập: thêm hai feature ngẫu nhiên để kiểm tra phương pháp

Theo ví dụ chính thức *Permutation Importance vs Random Forest Feature Importance (MDI)* của scikit-learn, ta thêm `random_num` (liên tục) và `random_cat` (phân loại, 3 mức). Cả hai **không liên quan** tới target. Phương pháp tốt phải xếp chúng gần 0.

```python
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from churn.data.split import time_split
from churn.data.synthetic import make_churn_data
from churn.features.clean import clean_churn

pd.set_option("display.width", 160)
rng = np.random.default_rng(0)
df = clean_churn(make_churn_data(n=20_000, seed=42))
df["random_num"] = rng.normal(size=len(df))
df["random_cat"] = rng.choice(["a", "b", "c"], size=len(df))
num = ["age", "tenure_months", "monthly_charges", "support_calls", "data_usage_gb", "random_num"]
cat = ["contract", "payment_method", "region", "random_cat"]
features = num + cat
train_df, valid_df, test_df = time_split(df, "signup_date", "2023-10-01", "2024-01-01")
X_train, y_train = train_df[features], train_df["churn"]
X_test, y_test = pd.concat([valid_df, test_df])[features], pd.concat([valid_df, test_df])["churn"]

prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                          OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), cat),
], verbose_feature_names_out=False).set_output(transform="pandas")

rf = Pipeline([("prep", prep), ("clf", RandomForestClassifier(
    n_estimators=200, min_samples_leaf=1, n_jobs=-1, random_state=0))]).fit(X_train, y_train)
gbm = Pipeline([("prep", prep), ("clf", lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, verbose=-1,
    random_state=0))]).fit(X_train, y_train)
for name, m in [("RandomForest (lá sâu, overfit)", rf), ("LightGBM", gbm)]:
    print(f"{name:30s} AUC train={roc_auc_score(y_train, m.predict_proba(X_train)[:, 1]):.3f} "
          f"test={roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]):.3f}")
```

## 10.2. Feature importance: MDI và permutation

**[Docs]** scikit-learn, *Permutation feature importance* và *Relation to impurity-based importance in trees*:

- **MDI** (mean decrease in impurity, `feature_importances_`) được tính trên **dữ liệu train**, nên có thể gán importance cao cho feature **không dự đoán được trên dữ liệu mới** khi mô hình overfit. MDI cũng **thiên vị mạnh** feature có **nhiều giá trị** (biến liên tục) so với biến nhị phân hay phân loại ít mức.
- **Permutation importance** đo mức **giảm điểm** khi xáo trộn một cột, tính được trên **dữ liệu held-out** và với **bất kỳ metric** nào. Lưu ý của tài liệu: feature ít quan trọng với một mô hình *tồi* có thể rất quan trọng với một mô hình *tốt*. Vì vậy hãy đánh giá mô hình trước, rồi mới tính importance.

$$i_j = s - \frac{1}{K}\sum_{k=1}^{K} s_{k,j}$$

```python
mdi = pd.Series(rf["clf"].feature_importances_, index=rf["prep"].get_feature_names_out())
# rf đã tự song song khi predict (n_jobs=-1) -> permutation_importance chạy tuần tự để tránh song song lồng nhau
perm_train = permutation_importance(rf, X_train, y_train, scoring="average_precision", n_repeats=5, random_state=0)
perm_test = permutation_importance(rf, X_test, y_test, scoring="average_precision", n_repeats=5, random_state=0)
cmp_imp = pd.DataFrame({"MDI (train)": mdi,
                        "perm (train)": pd.Series(perm_train.importances_mean, index=features),
                        "perm (test)": pd.Series(perm_test.importances_mean, index=features),
                        "perm_test_std": pd.Series(perm_test.importances_std, index=features)})
print(cmp_imp.sort_values("perm (test)", ascending=False).round(4))
```

**Đọc kết quả:** MDI xếp `random_num` và `data_usage_gb` (liên tục, không có tác động thật) **cao hơn** `contract` (3 mức, tác động mạnh). Permutation importance trên **test** đưa các biến ngẫu nhiên về ≈ 0. Khoảng cách giữa permutation trên train và trên test cho thấy mức độ overfit theo từng feature.

### Feature tương quan làm importance bị "chia nhỏ"

**[Docs]** *Misleading values on strongly correlated features*: khi xáo trộn một feature, mô hình vẫn lấy được thông tin qua feature tương quan với nó. Kết quả là **cả hai** đều có importance thấp, dù thông tin chung rất quan trọng. Tài liệu đề xuất **phân cụm các feature tương quan và giữ một đại diện** mỗi cụm (Chương 3.7).

```python
Xc_train = X_train.assign(tenure_copy=X_train["tenure_months"] + rng.normal(0, 0.5, len(X_train)))
Xc_test = X_test.assign(tenure_copy=X_test["tenure_months"] + rng.normal(0, 0.5, len(X_test)))
prep_c = ColumnTransformer([("num", SimpleImputer(strategy="median"), num + ["tenure_copy"]),
                            ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                                                  OrdinalEncoder(handle_unknown="use_encoded_value",
                                                                 unknown_value=-1)), cat)])
gbm_c = make_pipeline(prep_c, lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                                 min_child_samples=50, verbose=-1, random_state=0)).fit(Xc_train, y_train)
pi_c = permutation_importance(gbm_c, Xc_test, y_test, scoring="average_precision", n_repeats=5, random_state=0)
pi_o = permutation_importance(gbm, X_test, y_test, scoring="average_precision", n_repeats=5, random_state=0)
print("tenure (không có bản sao):", round(pd.Series(pi_o.importances_mean, index=features)["tenure_months"], 4))
print("tenure + bản sao        :", pd.Series(pi_c.importances_mean, index=list(Xc_test.columns))
      [["tenure_months", "tenure_copy"]].round(4).to_dict())
```

## 10.3. Partial Dependence (PDP), ICE và hiệu ứng tương tác

**[Docs]** *Partial Dependence and Individual Conditional Expectation plots*:

$$\text{PD}_S(x_S) = \mathbb{E}_{X_C}\big[f(x_S, X_C)\big] \approx \frac{1}{n}\sum_{i=1}^n f\big(x_S, x_C^{(i)}\big)$$

- **PDP** là hiệu ứng *trung bình* của feature lên dự đoán. **ICE** vẽ một đường cho *mỗi* mẫu. ICE tỏa ra nhiều nghĩa là có tương tác hoặc dị biệt.
- `centered=True` đưa mọi đường về cùng điểm xuất phát, giúp so sánh độ dốc.
- **Giả định độc lập:** PDP thay giá trị $x_S$ cho mọi mẫu, kể cả khi tổ hợp đó phi thực tế (tenure = 1 nhưng total_charges rất lớn). Khi feature tương quan mạnh, dùng **ALE** (Accumulated Local Effects; Apley & Zhu, 2020).
- Mặc định với cây/GBM của scikit-learn, `method="recursion"` rất nhanh nhưng chỉ cho `kind="average"`. ICE yêu cầu `method="brute"`.

```python
from sklearn.inspection import PartialDependenceDisplay

# scikit-learn 1.9 từ chối PDP trên cột kiểu số nguyên (lưới giá trị bị làm tròn ngầm) -> ép sang float
int_cols = X_test.select_dtypes("integer").columns
sample = X_test.sample(1500, random_state=0).astype({c: "float64" for c in int_cols})
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
PartialDependenceDisplay.from_estimator(
    gbm, sample, features=["support_calls", "tenure_months", "monthly_charges"], kind="both",
    subsample=150, centered=True, response_method="predict_proba", random_state=0, ax=axes,
    ice_lines_kw={"alpha": 0.15}, pd_line_kw={"color": "crimson", "lw": 2.5})
fig.suptitle("PDP (đỏ) + ICE (xám): support_calls tăng rủi ro, tenure giảm rủi ro")
plt.tight_layout()

fig, ax = plt.subplots(figsize=(6, 5))
PartialDependenceDisplay.from_estimator(gbm, sample, features=[("tenure_months", "support_calls")],
                                        kind="average", ax=ax)
ax.set_title("PDP 2 chiều: tương tác tenure × support_calls")
```

## 10.4. SHAP: Shapley values cho từng dự đoán

### Lý thuyết

Lý thuyết trò chơi hợp tác (Shapley, 1953) phân chia "phần thưởng" (dự đoán) cho các "người chơi" (feature) một cách **duy nhất** thỏa bốn tiên đề: *efficiency*, *symmetry*, *dummy*, *additivity*:

$$\phi_j = \sum_{S \subseteq F\setminus\{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\Big[v(S\cup\{j\}) - v(S)\Big]$$

SHAP (Lundberg & Lee, 2017) diễn giải mỗi dự đoán theo dạng **cộng tính**: $f(x) = \phi_0 + \sum_j \phi_j$, với $\phi_0$ là giá trị kỳ vọng (base value).

| Explainer | Mô hình | Ghi chú |
|---|---|---|
| **TreeExplainer** | Cây, RF, XGBoost, LightGBM, CatBoost | Chính xác, nhanh (đa thức). `feature_perturbation="tree_path_dependent"` (không cần background) hoặc `"interventional"` (cần background, đúng nghĩa "can thiệp" hơn) |
| LinearExplainer | Tuyến tính | Có xét tương quan feature |
| KernelExplainer | Bất kỳ | Model-agnostic nhưng chậm |
| Permutation / Exact | Bất kỳ | `shap.Explainer` tự chọn |
| Deep/GradientExplainer | Neural network | |

**Thang đo:** với LightGBM nhị phân, SHAP mặc định nằm trên **thang log-odds** (raw margin). Cộng base value với tổng SHAP ra đúng logit của dự đoán.

```python
import shap

X_test_t = gbm["prep"].transform(X_test)
explainer = shap.TreeExplainer(gbm["clf"])
sv = explainer(X_test_t.iloc[:2000])
values = sv.values[..., 1] if sv.values.ndim == 3 else sv.values          # một số phiên bản trả 2 lớp
base = np.ravel(sv.base_values)[0] if np.ndim(sv.base_values) else sv.base_values

# Kiểm tra tính cộng (efficiency): base + Σφ = logit(dự đoán)
raw_margin = gbm["clf"].predict_proba(X_test_t.iloc[:2000], raw_score=True)
print("Sai số cộng tính lớn nhất:", float(np.abs(base + values.sum(axis=1) - raw_margin).max()))

global_shap = pd.Series(np.abs(values).mean(axis=0), index=X_test_t.columns).sort_values(ascending=False)
print("mean |SHAP| (log-odds):")
print(global_shap.round(4))
```

> **Đối chiếu ground truth (Chương 0.8):** `contract`, `support_calls`, `tenure_months` có |SHAP| lớn nhất. `region`, `data_usage_gb`, `random_num`, `random_cat` gần 0. SHAP tìm đúng các yếu tố thật.

### Biểu đồ SHAP chuẩn

```python
plt.figure()
shap.plots.beeswarm(shap.Explanation(values, base_values=np.full(len(values), base), data=X_test_t.iloc[:2000].values,
                                     feature_names=list(X_test_t.columns)), max_display=10, show=False)
plt.title("Beeswarm: độ lớn + chiều tác động")
plt.tight_layout()

plt.figure()
shap.plots.scatter(shap.Explanation(values[:, list(X_test_t.columns).index("support_calls")],
                                    data=X_test_t["support_calls"].iloc[:2000].values,
                                    feature_names="support_calls"), show=False)
plt.title("Dependence: SHAP theo support_calls")
```

### Reason codes cho từng khách hàng

```python
READABLE = {"support_calls": "Gọi tổng đài hỗ trợ nhiều lần", "tenure_months": "Thời gian gắn bó",
            "contract": "Loại hợp đồng", "monthly_charges": "Mức cước tháng", "payment_method": "Phương thức thanh toán",
            "age": "Độ tuổi", "data_usage_gb": "Dung lượng sử dụng", "region": "Khu vực"}


def reason_codes(row_values: np.ndarray, row_data: pd.Series, k: int = 3) -> list[str]:
    contrib = pd.Series(row_values, index=row_data.index)
    top = contrib[contrib > 0].nlargest(k)
    return [f"{READABLE.get(f, f)} (giá trị={row_data[f]:.0f}, +{v:.2f} log-odds)" for f, v in top.items()]


proba = gbm["clf"].predict_proba(X_test_t.iloc[:2000])[:, 1]
i = int(np.argmax(proba))
print(f"Khách có rủi ro cao nhất: P(churn)={proba[i]:.2f}")
for r in reason_codes(values[i], X_test_t.iloc[i]):
    print("  -", r)
```

**[Kinh nghiệm]** Khi đưa reason codes ra cho người dùng cuối:

- Chỉ hiển thị **feature hành động được** hoặc dễ hiểu. Ẩn các feature kỹ thuật (cờ missing, ID mã hóa).
- Dùng ngôn ngữ nghiệp vụ, không dùng "log-odds".
- Kiểm tra **độ ổn định**: hai khách gần giống nhau phải có lý do gần giống nhau.

## 10.5. Mô hình tự giải thích được (glass-box) và mô hình thay thế

### Hệ số Logistic chuẩn hóa: odds ratio trên 1 độ lệch chuẩn

```python
from sklearn.linear_model import LogisticRegression

from churn.features.build import RAW_FEATURES, build_preprocessor

logit = make_pipeline(build_preprocessor(scale=True), LogisticRegression(max_iter=3000)).fit(
    train_df[RAW_FEATURES], y_train)
coef = pd.Series(logit[-1].coef_[0], index=logit[0].get_feature_names_out())
print(pd.DataFrame({"coef_per_sd": coef, "odds_ratio_per_sd": np.exp(coef)})
      .reindex(coef.abs().sort_values(ascending=False).index).head(8).round(3))
# Lưu ý: one-hot đầy đủ (không drop) + L2 -> hệ số các mức của cùng một biến chỉ có nghĩa TƯƠNG ĐỐI với nhau
# (two-year so với month-to-month), không diễn giải từng hệ số riêng lẻ.
```

### Global surrogate: cây nông bắt chước mô hình hộp đen

```python
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, export_text

surrogate = DecisionTreeRegressor(max_depth=3, min_samples_leaf=200, random_state=0)
target_logit = gbm["clf"].predict_proba(X_test_t, raw_score=True)
surrogate.fit(X_test_t, target_logit)
print(f"Độ trung thực (R² surrogate vs mô hình) = {r2_score(target_logit, surrogate.predict(X_test_t)):.3f}")
print(export_text(surrogate, feature_names=list(X_test_t.columns), decimals=1))
```

Surrogate chỉ đáng tin khi **độ trung thực (fidelity)** cao. Nó giải thích *mô hình*, không giải thích *dữ liệu*.

**Explainable Boosting Machine** (EBM, InterpretML của Microsoft) là GAM được huấn luyện bằng boosting, có thêm một số tương tác cặp. Độ chính xác thường gần GBM và **mỗi feature có đồ thị hiệu ứng riêng**. Đây là lựa chọn tốt khi cần cả hiệu năng lẫn minh bạch (ngân hàng, y tế).

```python norun
from interpret.glassbox import ExplainableBoostingClassifier

ebm = ExplainableBoostingClassifier(interactions=10, random_state=0).fit(X_train, y_train)
ebm.explain_global().visualize()        # đồ thị hiệu ứng từng feature
```

## 10.6. Fairness: công bằng giữa các nhóm

| Tiêu chí | Định nghĩa | Ý nghĩa | fairlearn |
|---|---|---|---|
| Demographic parity | $P(\hat Y=1\mid A=a)$ bằng nhau | Tỷ lệ được chọn như nhau | `demographic_parity_difference` |
| Equal opportunity | TPR bằng nhau | Người "xứng đáng" có cơ hội như nhau | `true_positive_rate` theo nhóm |
| Equalized odds | TPR **và** FPR bằng nhau | | `equalized_odds_difference` |
| Predictive parity | Precision bằng nhau | Dự đoán positive đáng tin như nhau | `MetricFrame` + precision |
| Calibration theo nhóm | Xác suất calibrate trong từng nhóm | | Reliability diagram theo nhóm |

> **Định lý bất khả thi** (Kleinberg, Mullainathan & Raghavan, 2016; Chouldechova, 2017): khi tỷ lệ cơ sở (base rate) khác nhau giữa các nhóm, **không thể** đồng thời thỏa calibration theo nhóm và equalized odds, trừ trường hợp tầm thường. Chọn tiêu chí là **quyết định nghiệp vụ/đạo đức/pháp lý**, cần stakeholder tham gia.

```python
from fairlearn.metrics import (MetricFrame, count, demographic_parity_difference, equalized_odds_difference,
                               false_positive_rate, selection_rate, true_positive_rate)
from sklearn.metrics import precision_score

threshold = 0.25
p_test = gbm.predict_proba(X_test)[:, 1]
pred = (p_test >= threshold).astype(int)


def age_bands(ages: pd.Series) -> pd.Series:
    """Nhóm tuổi; tuổi thiếu -> 'unknown'. Lưu ý pandas 3: .astype(str) GIỮ NaN là NaN (không thành 'nan'),
    nên phải gán nhãn tường minh, nếu không nhóm thiếu tuổi bị loại âm thầm khỏi phân tích fairness."""
    bands = pd.cut(ages, bins=[0, 30, 45, 60, 120], labels=["<30", "30-45", "45-60", "60+"])
    return bands.cat.add_categories("unknown").fillna("unknown").astype(str)


age_group = age_bands(X_test["age"])

mf = MetricFrame(metrics={"n": count, "base_rate": lambda y, p: np.mean(y), "selection_rate": selection_rate,
                          "TPR": true_positive_rate, "FPR": false_positive_rate,
                          "precision": lambda y, p: precision_score(y, p, zero_division=0)},
                 y_true=y_test, y_pred=pred, sensitive_features=age_group)
print(mf.by_group.round(3))
print("Demographic parity difference:", round(demographic_parity_difference(y_test, pred, sensitive_features=age_group), 3))
print("Equalized odds difference    :", round(equalized_odds_difference(y_test, pred, sensitive_features=age_group), 3))
```

### Giảm thiểu bias

| Giai đoạn | Kỹ thuật | Công cụ |
|---|---|---|
| Pre-processing | Cân bằng lại dữ liệu, loại **biến proxy** (mã vùng chi tiết có thể là proxy cho dân tộc/thu nhập) | `fairlearn.preprocessing.CorrelationRemover` |
| In-processing | Huấn luyện có ràng buộc fairness | `fairlearn.reductions.ExponentiatedGradient` |
| Post-processing | Ngưỡng khác nhau theo nhóm | `fairlearn.postprocessing.ThresholdOptimizer` |

```python
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.frozen import FrozenEstimator

# fairlearn không nhận NaN trong X. Điền NaN bằng đúng giá trị mà SimpleImputer của pipeline đã học
# (median cho số, "missing" cho phân loại) -> dự đoán của mô hình KHÔNG đổi.
fill = {**dict(zip(num, gbm["prep"].named_transformers_["num"].statistics_)), **{c: "missing" for c in cat}}


def no_nan(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[features].fillna(fill)


assert np.allclose(gbm.predict_proba(no_nan(valid_df))[:, 1], gbm.predict_proba(valid_df[features])[:, 1])

# Fit bộ hậu xử lý trên VALID (độc lập với dữ liệu train mô hình), đánh giá trên TEST
X_fit, y_fit = no_nan(valid_df), valid_df["churn"]
postproc = ThresholdOptimizer(estimator=FrozenEstimator(gbm), constraints="true_positive_rate_parity",
                              objective="balanced_accuracy_score", predict_method="predict_proba", prefit=True)
postproc.fit(X_fit, y_fit, sensitive_features=age_bands(valid_df["age"]))      # tuổi gốc (có thể thiếu)

age_te = age_bands(test_df["age"])
before = (gbm.predict_proba(test_df[features])[:, 1] >= threshold).astype(int)
after = postproc.predict(no_nan(test_df), sensitive_features=age_te, random_state=0)
for name, pr in [("trước", before), ("sau ThresholdOptimizer", after)]:
    tpr = MetricFrame(metrics=true_positive_rate, y_true=test_df["churn"], y_pred=pr, sensitive_features=age_te)
    print(f"TPR theo nhóm {name:22s}: {tpr.by_group.round(3).to_dict()} | chênh lệch={tpr.difference():.3f}")
```

> **[Kinh nghiệm]** Dùng ngưỡng khác nhau theo nhóm (post-processing) yêu cầu **dùng thuộc tính nhạy cảm lúc dự đoán**. Điều này có thể bị luật cấm trong một số lĩnh vực (tín dụng, tuyển dụng). Hãy tham vấn pháp chế trước khi áp dụng.

## 10.7. Model Card: tài liệu hóa mô hình

Theo Mitchell et al. (2019), *Model Cards for Model Reporting*:

```markdown
# Model Card: churn-classifier v3
## Chi tiết mô hình
- Loại: LightGBM (300 cây), pipeline tiền xử lý `churn.features.build`; MLflow model_id m-xxxx
- Chủ sở hữu: Data Science Team (ds-team@company.vn); ngày: 2026-10
## Mục đích sử dụng
- Dùng cho: xếp hạng khách có nguy cơ rời bỏ trong 30 ngày để CSKH liên hệ.
- KHÔNG dùng cho: từ chối dịch vụ, định giá cá nhân hóa, quyết định tín dụng.
## Dữ liệu
- Huấn luyện: khách đăng ký 2022-01 → 2023-09; đánh giá out-of-time 2023-10 → 2024-06.
- Nhãn: hủy dịch vụ trong 30 ngày sau ngày chấm điểm (gap 7 ngày).
## Hiệu năng (test out-of-time, 95% CI bootstrap)
- PR-AUC 0.39 [0.36–0.42]; ROC-AUC 0.75 [0.73–0.77]; ngưỡng 0.33 (tối đa lợi nhuận kỳ vọng).
## Phân tích theo nhóm & fairness
- Chênh lệch TPR giữa các nhóm tuổi ≤ 0.05 sau hậu xử lý; kém chính xác hơn với khách < 3 tháng.
## Giải thích
- Yếu tố chính: loại hợp đồng, số cuộc gọi hỗ trợ, thời gian gắn bó (mean |SHAP|).
## Hạn chế & rủi ro
- Chưa kiểm định cho khách doanh nghiệp; nhạy với thay đổi chính sách giá (theo dõi drift).
## Giám sát
- PSI feature/score hằng tuần; retrain khi PR-AUC giảm > 10% hoặc PSI > 0.25.
```

Tài liệu bổ sung cho dữ liệu: **Datasheets for Datasets** (Gebru et al., 2021).

> **Checklist Chương 10**
> - [ ] Đánh giá mô hình trước, giải thích sau. Importance tính trên **held-out** (permutation/SHAP), không chỉ MDI.
> - [ ] Kiểm tra phương pháp bằng biến ngẫu nhiên; xử lý feature tương quan (gom cụm) trước khi diễn giải.
> - [ ] Hướng và hình dạng tác động (PDP/ICE/SHAP) hợp lý về nghiệp vụ; cân nhắc monotonic constraints.
> - [ ] Reason codes dùng ngôn ngữ nghiệp vụ, chỉ gồm feature hiểu được, có kiểm tra độ ổn định.
> - [ ] Đo fairness theo các nhóm liên quan; chọn tiêu chí cùng stakeholder; tham vấn pháp chế về giảm thiểu.
> - [ ] Có Model Card ghi rõ mục đích, phạm vi, hiệu năng theo nhóm, hạn chế, chủ sở hữu.

### Tài liệu tham khảo Chương 10

- scikit-learn User Guide: *Inspection* (*Partial Dependence and ICE plots*; *Permutation feature importance*: Outline, Relation to impurity-based importance, Misleading values on strongly correlated features).
- scikit-learn Examples: *Permutation Importance vs Random Forest Feature Importance (MDI)*; *Permutation Importance with Multicollinear or Correlated Features*; *Common pitfalls in the interpretation of coefficients of linear models*.
- SHAP documentation: *An introduction to explainable AI with Shapley values*; API `TreeExplainer`. shap.readthedocs.io
- Lundberg, S. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS. · Lundberg, S. et al. (2020). *From local explanations to global understanding with explainable AI for trees.* Nature MI 2.
- Molnar, C. (2022). *Interpretable Machine Learning*, 2nd ed. christophm.github.io/interpretable-ml-book
- Apley, D. & Zhu, J. (2020). *Visualizing the effects of predictor variables in black box supervised learning models (ALE).* JRSS-B 82(4).
- Breiman, L. (2001). *Random Forests.* Machine Learning 45. · Strobl, C. et al. (2007). *Bias in random forest variable importance measures.* BMC Bioinformatics 8.
- fairlearn User Guide: *Assessment* (MetricFrame), *Mitigation* (ThresholdOptimizer, ExponentiatedGradient). fairlearn.org
- Kleinberg, J., Mullainathan, S. & Raghavan, M. (2016). *Inherent Trade-Offs in the Fair Determination of Risk Scores.* · Chouldechova, A. (2017). *Fair prediction with disparate impact.* Big Data 5(2).
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting.* FAT\*. · Gebru, T. et al. (2021). *Datasheets for Datasets.* CACM.
- Nori, H. et al. (2019). *InterpretML: A Unified Framework for Machine Learning Interpretability.*
