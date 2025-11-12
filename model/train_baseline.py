# model/train_baseline_v2.py
"""
Treino v2 (baseline melhorado) — usa features derivadas (recência, frequência, avg value, pct active)
e split temporal. Evita vazamento e salva artefatos em model/artifacts.

Rode:
    .venv\Scripts\Activate
    python model\train_baseline_v2.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42

# Paths
repo_root = Path(__file__).resolve().parent.parent
data_path = repo_root / "data" / "training_dataset.parquet"
model_dir = repo_root / "model" / "artifacts"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "churn_baseline_v2.pkl"
features_path = model_dir / "feature_list_v2.json"

# ------------- load dataset -------------
if not data_path.exists():
    raise SystemExit(f"Arquivo não encontrado: {data_path}. Gere com model/build_training_dataset.py")

df = pd.read_parquet(data_path)
print("Loaded dataset:", data_path, "shape:", df.shape)
print("Columns:", df.columns.tolist())

# ------------- label detection -------------
label_candidates = ["label", "churn", "target"]
label_col = next((c for c in label_candidates if c in df.columns), None)
if label_col is None:
    raise SystemExit("Não foi encontrada coluna de label no dataset. Coloque 'label' ou ajuste o script.")

# ------------- candidate engineered features -------------
# prioridade: use as melhores features derivadas se presentes
candidate_features_priority = [
    "num_orders_7d",
    "num_orders_30d",
    "avg_order_value_7d",
    "avg_order_value_30d",
    "days_since_last_order",
    "pct_active_60d",
    # fallback: snapshot/features from Feast
    "orders_30d",
    "avg_order_value",
]

# detect which of these exist in df
available = [c for c in candidate_features_priority if c in df.columns]
print("Available candidate derived features:", available)

# choose features — prefer derived windows over snapshot columns to avoid leakage
# but still keep robust fallback list
selected_features = []
for c in candidate_features_priority:
    if c in df.columns:
        # Don't include orders_30d if it was used to build the label (leakage check)
        if c == "orders_30d":
            # check if label equals (orders_30d == 0)
            try:
                leak_check = (df["orders_30d"].fillna(0) == 0).astype(int)
                if label_col in df.columns and leak_check.equals(df[label_col].astype(int)):
                    print("Detected leakage: 'orders_30d' reproduces the label. Skipping it as feature.")
                    continue
            except Exception:
                # if any problem in test, skip using orders_30d
                print("Warning: could not fully validate orders_30d leakage; skipping it for safety.")
                continue
        selected_features.append(c)

if not selected_features:
    raise SystemExit("Nenhuma feature derivada detectada. Rode build_training_dataset.py corretamente.")

print("Selected features for training:", selected_features)

# ------------- ensure timestamps and temporal split -------------
if "event_timestamp" not in df.columns:
    raise SystemExit("Necessário 'event_timestamp' no dataset para split temporal. Ajuste build script.")

df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
df = df.sort_values("event_timestamp").reset_index(drop=True)

# temporal split: 80% time for training, 20% for validation
cutoff = df["event_timestamp"].quantile(0.8)
train_df = df[df["event_timestamp"] <= cutoff].copy()
val_df = df[df["event_timestamp"] > cutoff].copy()
print("Temporal split cutoff:", cutoff)
print("Train rows:", train_df.shape[0], "Val rows:", val_df.shape[0])

# ------------- prepare X/y -------------
X_train = train_df[selected_features].copy()
X_val = val_df[selected_features].copy()
y_train = train_df[label_col].astype(int).copy()
y_val = val_df[label_col].astype(int).copy()

# types: ensure numeric (days_since_last_order may be float/int/NaN)
for col in X_train.columns:
    if not pd.api.types.is_numeric_dtype(X_train[col]):
        # try to parse datetimes to numeric (e.g., last_order_ts accidentally present) or drop
        try:
            X_train[col] = pd.to_numeric(X_train[col])
            X_val[col] = pd.to_numeric(X_val[col])
        except Exception:
            # if cannot convert, drop this feature
            print(f"Warning: feature {col} is not numeric and could not be coerced. Dropping.")
            X_train.drop(columns=[col], inplace=True, errors=True)
            X_val.drop(columns=[col], inplace=True, errors=True)

# re-evaluate selected features after coercion
selected_features = [c for c in selected_features if c in X_train.columns]
if not selected_features:
    raise SystemExit("After coercion no features left for training. Aborting.")

# impute missing: numeric -> median for avg values, 0 for counts
for c in selected_features:
    if c.startswith("num_") or c.startswith("pct_"):
        fill_value = 0
    else:
        fill_value = X_train[c].median() if not X_train[c].dropna().empty else 0
    X_train[c] = X_train[c].fillna(fill_value)
    X_val[c] = X_val[c].fillna(fill_value)

# final numpy conversion
X_train = X_train.astype(float)
X_val = X_val.astype(float)

# --- Remove colunas com variância baixa (quase tudo zero) ---
low_var = [c for c in selected_features
           if (X_train[c].std() < 1e-6) or ((X_train[c]==0).mean() > 0.9)]
if low_var:
    print("Dropping low-variance features:", low_var)
    for c in low_var:
        selected_features.remove(c)
        X_train.drop(columns=[c], inplace=True, errors=True)
        X_val.drop(columns=[c], inplace=True, errors=True)

print("Final feature set:", selected_features)
print("X_train shape:", X_train.shape, "X_val shape:", X_val.shape)
print("Label distribution (train):\n", y_train.value_counts())
print("Label distribution (val):\n", y_val.value_counts())


scaler = RobustScaler()
X_train.loc[:, :] = scaler.fit_transform(X_train)
X_val.loc[:, :] = scaler.transform(X_val)
# ------------- handle class imbalance -------------
pos = int(y_train.sum())
neg = int(len(y_train) - pos)
scale_pos_weight = (neg / (pos + 1e-9)) if pos > 0 else 1.0
print("scale_pos_weight:", scale_pos_weight)

# --- Remover features que reproduzem o label (vazamento) ---
leaky = []
for c in selected_features.copy():
    # compara ranges — se max_neg < min_pos ou max_pos < min_neg significa separador perfeito
    neg = df[df[label_col]==0][c].dropna()
    pos = df[df[label_col]==1][c].dropna()
    if len(neg) == 0 or len(pos) == 0:
        continue
    if neg.max() < pos.min() or pos.max() < neg.min():
        leaky.append(c)

if leaky:
    print("Removendo features que reproduzem o label (vazamento):", leaky)
    for c in leaky:
        if c in selected_features:
            selected_features.remove(c)
            X_train.drop(columns=[c], inplace=True, errors=True)
            X_val.drop(columns=[c], inplace=True, errors=True)

# ------------- LightGBM training (compatible with >=4.0) -------------
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "seed": RANDOM_STATE,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "scale_pos_weight": scale_pos_weight,
}

print("LightGBM version:", lgb.__version__)
version = tuple(int(x) for x in lgb.__version__.split(".")[:2])

if version >= (4, 0):
    bst = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
    )
else:
    bst = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=50,
    )

# ------------- Evaluation -------------
preds_val = bst.predict(X_val, num_iteration=getattr(bst, "best_iteration", None))
auc = roc_auc_score(y_val, preds_val) if len(np.unique(y_val)) > 1 else float("nan")
preds_label = (preds_val >= 0.5).astype(int)
acc = accuracy_score(y_val, preds_label)

print(f"\nValidation AUC: {auc:.4f}")
print(f"Validation Accuracy (0.5 cutoff): {acc:.4f}")
print("\nClassification report (val):")
print(classification_report(y_val, preds_label, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_val, preds_label))

# ------------- Feature importance -------------
try:
    fi = pd.DataFrame({"feature": bst.feature_name(), "importance": bst.feature_importance()})
    print("\nFeature importance:\n", fi.sort_values("importance", ascending=False))
except Exception as e:
    print("Could not extract feature importance:", e)

# ------------- Save artifacts -------------
joblib.dump(bst, model_path)
with open(features_path, "w", encoding="utf8") as f:
    json.dump(selected_features, f, ensure_ascii=False, indent=2)

print("\nSaved model to:", model_path)
print("Saved feature list to:", features_path)
print("Best iteration:", getattr(bst, "best_iteration", None))

import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

probs = bst.predict(X_val)
prec, rec, thr = precision_recall_curve(y_val, probs)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1s)
best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
print(f"Melhor F1={f1s[best_idx]:.3f} com cutoff={best_thr:.2f}")

y_pred_best = (probs >= best_thr).astype(int)
print("Confusion matrix no melhor cutoff:")
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_val, y_pred_best))
