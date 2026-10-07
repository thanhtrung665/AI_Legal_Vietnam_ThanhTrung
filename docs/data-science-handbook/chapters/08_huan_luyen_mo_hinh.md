# CHƯƠNG 8. HUẤN LUYỆN MÔ HÌNH (MODEL TRAINING)

## 8.1. Lộ trình huấn luyện chuẩn

```text
1. Baseline ngây thơ  →  2. Baseline nghiệp vụ  →  3. Mô hình tuyến tính  →  4. GBM mặc định
        →  5. Tuning siêu tham số  →  6. Ensemble (nếu đáng)  →  7. Phân tích lỗi  →  lặp lại
```

Mỗi bước phải **vượt bước trước một cách có ý nghĩa thống kê** mới đáng giữ độ phức tạp tăng thêm.

## 8.2. Chọn thuật toán

| Dữ liệu | Lựa chọn đầu tiên | Thay thế |
|---|---|---|
| Bảng (tabular), < 1M dòng | **LightGBM / XGBoost / CatBoost** | Random Forest, Logistic Regression |
| Bảng, cần giải thích tuyệt đối (ngân hàng, bảo hiểm) | Logistic Regression + WoE (scorecard), EBM (`interpret`) | GAM, cây quyết định nông |
| Bảng, nhiều biến phân loại | **CatBoost** | LightGBM native categorical |
| Văn bản | Fine-tune Transformer (PhoBERT cho tiếng Việt) | TF-IDF + Logistic Regression (baseline mạnh) |
| Ảnh | CNN / ViT pretrained (transfer learning) | — |
| Chuỗi thời gian | GBM + lag/rolling features, ETS/ARIMA, Prophet | N-BEATS, TFT, foundation models (Chronos, TimesFM) |
| Rất ít dữ liệu (< 1000) | Mô hình tuyến tính có regularization, RF | Bayesian models |

### Bias – Variance trade-off

| Triệu chứng | Chẩn đoán | Hành động |
|---|---|---|
| Train thấp, Valid thấp | **Underfitting (high bias)** | Mô hình phức tạp hơn, thêm feature, giảm regularization |
| Train cao, Valid thấp hơn nhiều | **Overfitting (high variance)** | Thêm dữ liệu, regularization, giảm độ sâu, early stopping, bớt feature |
| Train cao, Valid cao, Test thấp | **Distribution shift / leakage trong valid** | Xem lại cách split, adversarial validation |

## 8.3. Baseline

```python
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

# (Tiếp tục từ chương 7: X_train, X_valid, y_train, y_valid đã có)

# Baseline 1: ngây thơ — dự đoán xác suất = tỷ lệ lớp
dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
p = dummy.predict_proba(X_valid)[:, 1]
print("Dummy  PR-AUC:", round(average_precision_score(y_valid, p), 4))   # = tỷ lệ positive


# Baseline 2: quy tắc nghiệp vụ hiện tại
def rule_based_score(X: pd.DataFrame) -> np.ndarray:
    return ((X["contract"] == "month-to-month").astype(int) * 2
            + (X["support_calls"] >= 3).astype(int)
            + (X["tenure_months"] < 6).astype(int)).to_numpy()


s = rule_based_score(X_valid)
print("Rule   ROC-AUC:", round(roc_auc_score(y_valid, s), 4),
      "PR-AUC:", round(average_precision_score(y_valid, s), 4))
```

## 8.4. So sánh nhiều mô hình bằng cùng một CV

```python
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn.features.build import build_preprocessor

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

candidates = {
    "logreg": Pipeline([("prep", build_preprocessor(scale=True)),
                        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000))]),
    "random_forest": Pipeline([("prep", build_preprocessor(scale=False)),
                               ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                                              class_weight="balanced_subsample",
                                                              n_jobs=-1, random_state=42))]),
    "hist_gb": Pipeline([("prep", build_preprocessor(scale=False)),
                         ("clf", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                                                early_stopping=True, random_state=42))]),
    "lightgbm": Pipeline([("prep", build_preprocessor(scale=False)),
                          ("clf", LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                                 scale_pos_weight=pos_weight, verbose=-1,
                                                 random_state=42))]),
}

rows = []
for name, model in candidates.items():
    r = cross_validate(model, X_train, y_train, cv=cv, n_jobs=-1,
                       scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision",
                                "neg_log_loss": "neg_log_loss"})
    rows.append({"model": name,
                 **{k.replace("test_", ""): f"{v.mean():.4f} ± {v.std():.4f}"
                    for k, v in r.items() if k.startswith("test_")},
                 "fit_time_s": round(r["fit_time"].mean(), 2)})
print(pd.DataFrame(rows).set_index("model"))
```

## 8.5. Gradient Boosting chuyên sâu

### Siêu tham số quan trọng của LightGBM

| Nhóm | Tham số | Ý nghĩa | Khoảng tìm kiếm điển hình |
|---|---|---|---|
| Tốc độ học | `learning_rate` | Bước học; nhỏ hơn + nhiều cây hơn = tốt hơn nhưng chậm | 0.01 – 0.1 |
| | `n_estimators` | Số cây — **dùng early stopping**, đừng tune trực tiếp | 100 – 10000 |
| Độ phức tạp | `num_leaves` | Số lá tối đa (tham số chính) | 15 – 255 |
| | `max_depth` | Giới hạn độ sâu | -1, 4 – 12 |
| | `min_child_samples` | Số mẫu tối thiểu/lá — chống overfit | 10 – 200 |
| Ngẫu nhiên | `subsample` (+ `subsample_freq=1`) | Lấy mẫu dòng | 0.5 – 1.0 |
| | `colsample_bytree` | Lấy mẫu cột | 0.4 – 1.0 |
| Regularization | `reg_alpha` (L1), `reg_lambda` (L2) | | 1e-8 – 10 (log) |
| Mất cân bằng | `scale_pos_weight` / `is_unbalance` | | neg/pos |

### Early stopping đúng cách

```python
import lightgbm as lgb

prep = build_preprocessor(scale=False)
Xtr = prep.fit_transform(X_train, y_train)     # fit preprocessor CHỈ trên train
Xva = prep.transform(X_valid)

model = lgb.LGBMClassifier(
    n_estimators=10_000, learning_rate=0.02, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    verbose=-1, random_state=42,
)
model.fit(
    Xtr, y_train,
    eval_set=[(Xva, y_valid)],
    eval_metric="average_precision",
    callbacks=[lgb.early_stopping(stopping_rounds=200, first_metric_only=True),
               lgb.log_evaluation(period=500)],
)
print("Best iteration:", model.best_iteration_)
```

> **Lưu ý:** Tập dùng cho early stopping đã "bị nhìn" → không dùng nó để báo cáo điểm cuối cùng. Khi train lại trên train+valid, đặt `n_estimators = best_iteration × (1 + tỷ lệ dữ liệu tăng thêm)` (ví dụ ×1.1–1.2).

### XGBoost & CatBoost

```python
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=10_000, learning_rate=0.02, max_depth=6, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
    eval_metric="aucpr", early_stopping_rounds=200, scale_pos_weight=pos_weight,
    random_state=42, n_jobs=-1,
)
xgb.fit(Xtr, y_train, eval_set=[(Xva, y_valid)], verbose=False)

# CatBoost: đưa thẳng cột phân loại dạng chuỗi, không cần encode
cat_cols = ["contract", "payment_method", "region"]
feature_cols = ["age", "tenure_months", "monthly_charges", "total_charges",
                "support_calls", "data_usage_gb", *cat_cols]


def to_catboost(X: pd.DataFrame) -> pd.DataFrame:
    X = X[feature_cols].copy()
    X[cat_cols] = X[cat_cols].fillna("missing").astype(str)
    return X


cb = CatBoostClassifier(
    iterations=10_000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
    eval_metric="PRAUC", auto_class_weights="Balanced",
    early_stopping_rounds=200, random_seed=42, verbose=0,
)
cb.fit(to_catboost(X_train), y_train, cat_features=cat_cols,
       eval_set=(to_catboost(X_valid), y_valid), use_best_model=True)
```

## 8.6. Tối ưu siêu tham số (Hyperparameter Optimization)

| Phương pháp | Ưu | Nhược | Công cụ |
|---|---|---|---|
| Grid Search | Đơn giản, toàn diện | Bùng nổ tổ hợp | `GridSearchCV` |
| Random Search | Hiệu quả hơn grid với nhiều tham số (Bergstra & Bengio, 2012) | Không học từ lần thử trước | `RandomizedSearchCV` |
| Successive Halving / Hyperband | Loại sớm cấu hình kém | | `HalvingRandomSearchCV` |
| **Bayesian (TPE, GP)** | Học từ lịch sử, ít lần thử | Tuần tự hơn | **Optuna**, Hyperopt, scikit-optimize |
| Population-based | Tốt cho deep learning | Tốn tài nguyên | Ray Tune |

### Random Search (baseline cho tuning)

```python
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    candidates["lightgbm"],
    param_distributions={
        "clf__num_leaves": randint(15, 128),
        "clf__min_child_samples": randint(10, 200),
        "clf__learning_rate": loguniform(0.01, 0.1),
        "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__reg_lambda": loguniform(1e-3, 10),
    },
    n_iter=40, cv=cv, scoring="average_precision", n_jobs=-1, random_state=42, refit=True,
)
search.fit(X_train, y_train)
print(search.best_score_, search.best_params_)
```

### Optuna — Bayesian Optimization với pruning (khuyến nghị)

```python
import optuna
from sklearn.base import clone
from sklearn.metrics import average_precision_score


def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, pos_weight),
        "verbose": -1, "random_state": 42,
    }
    scores = []
    for fold, (tr, va) in enumerate(cv.split(X_train, y_train)):
        prep = build_preprocessor(scale=False)
        Xtr_f = prep.fit_transform(X_train.iloc[tr], y_train.iloc[tr])
        Xva_f = prep.transform(X_train.iloc[va])
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr_f, y_train.iloc[tr], eval_set=[(Xva_f, y_train.iloc[va])],
              eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
        scores.append(average_precision_score(y_train.iloc[va], m.predict_proba(Xva_f)[:, 1]))
        trial.set_user_attr(f"best_iter_fold{fold}", m.best_iteration_)
        trial.report(float(np.mean(scores)), step=fold)
        if trial.should_prune():               # dừng sớm trial kém sau vài fold
            raise optuna.TrialPruned()
    return float(np.mean(scores))


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    study_name="churn-lgbm",
    storage="sqlite:///optuna.db", load_if_exists=True,   # lưu lịch sử, tiếp tục được
)
study.optimize(objective, n_trials=100, timeout=3600, show_progress_bar=True)
print("Best PR-AUC:", study.best_value)
print("Best params:", study.best_params)

# Phân tích: tham số nào quan trọng?
optuna.visualization.plot_param_importances(study).show()
optuna.visualization.plot_optimization_history(study).show()
```

> **Kinh nghiệm:** Tuning thường chỉ mang lại thêm 1–3% so với tham số mặc định hợp lý. Feature engineering và dữ liệu tốt mang lại nhiều hơn. Đừng dành 80% thời gian cho tuning.

## 8.7. Ensemble

| Kỹ thuật | Mô tả | Khi nào hiệu quả |
|---|---|---|
| Bagging | Trung bình nhiều mô hình trên mẫu bootstrap (Random Forest) | Giảm variance |
| Boosting | Mô hình sau sửa lỗi mô hình trước (GBM) | Giảm bias |
| Averaging / Voting | Trung bình xác suất nhiều mô hình khác loại | Mô hình đa dạng, tương quan lỗi thấp |
| Rank averaging | Trung bình thứ hạng thay vì xác suất | Metric dựa trên thứ hạng (AUC) |
| **Stacking** | Meta-model học trên OOF prediction | Thi đấu; production cân nhắc độ phức tạp |
| Seed averaging | Cùng mô hình, nhiều seed | Ổn định dự đoán, rẻ |

```python
from sklearn.ensemble import StackingClassifier, VotingClassifier

voting = VotingClassifier(
    estimators=[("lr", candidates["logreg"]), ("lgbm", candidates["lightgbm"]),
                ("rf", candidates["random_forest"])],
    voting="soft", weights=[1, 2, 1], n_jobs=-1,
)

stacking = StackingClassifier(
    estimators=[("lr", candidates["logreg"]), ("lgbm", candidates["lightgbm"]),
                ("rf", candidates["random_forest"])],
    final_estimator=LogisticRegression(C=1.0, max_iter=2000),
    cv=StratifiedKFold(5, shuffle=True, random_state=0),   # meta-model học trên OOF
    stack_method="predict_proba", n_jobs=-1,
)
```

> **Production trade-off:** Stacking 3 mô hình = 3× độ trễ, 3× bảo trì, khó giải thích. Chỉ dùng khi cải thiện có giá trị kinh doanh rõ ràng.

## 8.8. Chẩn đoán bằng Learning Curve & Validation Curve

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve

sizes, tr_scores, va_scores = learning_curve(
    candidates["lightgbm"], X_train, y_train, cv=cv, scoring="average_precision",
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1, shuffle=True, random_state=0,
)
plt.plot(sizes, tr_scores.mean(1), "o-", label="train")
plt.plot(sizes, va_scores.mean(1), "o-", label="validation")
plt.fill_between(sizes, va_scores.mean(1) - va_scores.std(1),
                 va_scores.mean(1) + va_scores.std(1), alpha=0.2)
plt.xlabel("Số mẫu train"); plt.ylabel("PR-AUC"); plt.legend(); plt.title("Learning curve")
# Validation còn tăng khi thêm dữ liệu -> thu thập thêm dữ liệu sẽ có lợi
# Hai đường hội tụ ở mức thấp -> underfitting: cần feature/mô hình tốt hơn

param_range = [7, 15, 31, 63, 127, 255]
tr_s, va_s = validation_curve(candidates["lightgbm"], X_train, y_train,
                              param_name="clf__num_leaves", param_range=param_range,
                              cv=cv, scoring="average_precision", n_jobs=-1)
```

## 8.9. Deep Learning cho dữ liệu bảng (PyTorch)

Với dữ liệu bảng, GBM thường thắng (Grinsztajn et al., 2022). Deep learning đáng thử khi: dữ liệu rất lớn, kết hợp đa phương thức (bảng + văn bản + ảnh), cần embedding cho biến cardinality cực cao, hoặc học online.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TabularMLP(nn.Module):
    def __init__(self, n_in: int, hidden=(256, 128), dropout: float = 0.2):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)          # logits


def train_mlp(Xtr, ytr, Xva, yva, epochs=100, patience=10, lr=1e-3, batch_size=512, seed=42):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    to_t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))
    train_dl = DataLoader(TensorDataset(to_t(Xtr), to_t(ytr)), batch_size=batch_size, shuffle=True)
    Xva_t = to_t(Xva).to(device)

    model = TabularMLP(Xtr.shape[1]).to(device)
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best, best_state, wait = -np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva_t)).cpu().numpy()
        score = average_precision_score(yva, p)
        sched.step(score)
        if score > best + 1e-4:
            best, wait = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch}, best PR-AUC={best:.4f}")
                break
    model.load_state_dict(best_state)
    return model


# Đầu vào phải được scale (build_preprocessor(scale=True))
prep_nn = build_preprocessor(scale=True)
Xtr_nn = prep_nn.fit_transform(X_train, y_train)
Xva_nn = prep_nn.transform(X_valid)
mlp = train_mlp(Xtr_nn, y_train.to_numpy(), Xva_nn, y_valid.to_numpy())
```

Các kiến trúc chuyên cho tabular: **TabNet**, **FT-Transformer**, **TabPFN** (rất mạnh với dữ liệu nhỏ < 10k dòng), **SAINT**.

## 8.10. Script huấn luyện production (kết nối MLflow)

```python
# src/churn/models/train.py
import argparse
import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from churn.config import Secrets, load_config
from churn.data.validate import validate
from churn.features.build import build_preprocessor
from churn.features.clean import clean_churn
from churn.data.split import time_split  # hàm ở chương 7

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(Secrets().mlflow_tracking_uri)
    mlflow.set_experiment("churn-prediction")

    df = clean_churn(pd.read_parquet("data/raw/churn.parquet"))
    validate(df)
    train_df, valid_df, test_df = time_split(df, "signup_date", "2023-12-01", "2024-03-01")
    X_tr, y_tr = train_df.drop(columns=cfg.target), train_df[cfg.target]
    X_va, y_va = valid_df.drop(columns=cfg.target), valid_df[cfg.target]

    with mlflow.start_run(run_name="lgbm") as run:
        mlflow.log_params(cfg.model_params)
        mlflow.log_params({"n_train": len(X_tr), "n_valid": len(X_va),
                           "train_churn_rate": round(y_tr.mean(), 4)})

        pipe = Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("clf", lgb.LGBMClassifier(**cfg.model_params, random_state=cfg.random_state, verbose=-1)),
        ])
        pipe.fit(X_tr, y_tr)

        proba = pipe.predict_proba(X_va)[:, 1]
        metrics = {"valid_roc_auc": roc_auc_score(y_va, proba),
                   "valid_pr_auc": average_precision_score(y_va, proba)}
        mlflow.log_metrics(metrics)
        log.info("Metrics: %s", metrics)

        Path("models").mkdir(exist_ok=True)
        joblib.dump(pipe, "models/model.joblib")
        Path("reports").mkdir(exist_ok=True)
        Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2))

        mlflow.sklearn.log_model(
            pipe, "model",
            signature=infer_signature(X_va.head(100), proba[:100]),
            input_example=X_va.head(5),
            registered_model_name="churn-classifier",
        )
        log.info("Run ID: %s", run.info.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    main(parser.parse_args().config)
```

```yaml
# configs/train.yaml
target: churn
random_state: 42
model_params:
  n_estimators: 800
  learning_rate: 0.03
  num_leaves: 31
  min_child_samples: 50
  subsample: 0.8
  subsample_freq: 1
  colsample_bytree: 0.8
  reg_lambda: 1.0
```

> **Checklist Chương 8**
> - [ ] Có baseline ngây thơ + baseline nghiệp vụ.
> - [ ] Mọi mô hình so sánh trên cùng CV, cùng metric, báo cáo mean ± std.
> - [ ] Early stopping dùng tập validation riêng, không phải test.
> - [ ] Tuning có giới hạn ngân sách, lưu lịch sử (Optuna storage), seed cố định.
> - [ ] Đã xem learning curve để quyết định: thêm dữ liệu hay cải thiện mô hình.
> - [ ] Toàn bộ pipeline (tiền xử lý + mô hình) được lưu thành 1 artifact và log vào MLflow.
