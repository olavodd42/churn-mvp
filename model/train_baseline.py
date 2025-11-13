#!/usr/bin/env python3
"""
model/train_baseline_v2.py — versão melhorada

- Split temporal sem overlap (gera cutoff por quantil).
- Remove features com variância quase-zero e >90% zeros.
- Detecta e remove features que reproduzem/descobrem o label (vazamento)
  via thresholding/monotonic check (empírico).
- Usa RobustScaler, parâmetros LightGBM mais conservadores (evita mensagens -inf).
- Salva: modelo, feature_list, scaler em model/artifacts/.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

RANDOM_STATE = 42

# Paths
repo_root = Path(__file__).resolve().parent.parent
data_path = repo_root / "data" / "training_dataset.parquet"
model_dir = repo_root / "model" / "artifacts"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "churn_baseline_v2.pkl"
features_path = model_dir / "feature_list_v2.json"
scaler_path = model_dir / "scaler_v2.pkl"

# ---------- util ----------
def is_almost_constant(s, zero_frac_thresh=0.9, std_thresh=1e-6):
    if s.dtype.kind in "if":  # numeric
        return (s == 0).mean() > zero_frac_thresh or (s.std() if not s.isna().all() else 0) < std_thresh
    return False

def find_threshold_predictor(series, labels, require_accuracy=0.995):
    """Tenta achar um threshold que separa label==1 por > threshold ou < threshold.
       Retorna (best_acc, best_thresh, direction) ou (None, None, None).
    """
    if series.dropna().empty:
        return (None, None, None)
    # se muito poucas uniq, fallback
    uniq = np.unique(series.dropna().values)
    if uniq.size <= 10:
        candidates = uniq
    else:
        candidates = np.percentile(series.dropna().values, np.linspace(1, 99, 99))
    best = (0.0, None, None)
    y = labels.values.astype(int)
    for t in np.unique(candidates):
        # predict 1 if value > t
        pred1 = (series.fillna(-1) > t).astype(int).values
        acc1 = (pred1 == y).mean()
        # predict 1 if value <= t
        pred2 = (series.fillna(-1) <= t).astype(int).values
        acc2 = (pred2 == y).mean()
        if acc1 > best[0]:
            best = (acc1, float(t), "gt")
        if acc2 > best[0]:
            best = (acc2, float(t), "le")
    if best[0] >= require_accuracy:
        return best
    return (None, None, None)

# ---------- main ----------
if __name__ == "__main__":
    if not data_path.exists():
        raise SystemExit(f"Arquivo não encontrado: {data_path}. Gere com model/build_training_dataset.py")

    df = pd.read_parquet(data_path)
    print("Loaded dataset:", data_path, "shape:", df.shape)

    # detect label
    label_col = next((c for c in ("label", "churn", "target") if c in df.columns), None)
    if label_col is None:
        raise SystemExit("Nenhuma coluna de label encontrada no dataset.")
    print("Detected label:", label_col)

    # ensure event_timestamp exists
    if "event_timestamp" not in df.columns:
        raise SystemExit("Necessário 'event_timestamp' para split temporal.")
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    df = df.sort_values("event_timestamp").reset_index(drop=True)

    # temporal split (no user overlap)
    cutoff = df["event_timestamp"].quantile(0.8)
    train_df = df[df["event_timestamp"] <= cutoff].copy()
    val_df = df[df["event_timestamp"] > cutoff].copy()
    print("Temporal split cutoff:", cutoff, "Train rows:", len(train_df), "Val rows:", len(val_df))

    # candidate features: automatically pick numeric non-id/timestamp columns
    exclude_prefixes = ("user_id", "event_timestamp", "event_ts", "last_order_ts", "timestamp")
    candidate_features = [c for c in df.columns if c not in (label_col,) and not any(c.startswith(p) for p in exclude_prefixes)]
    numeric_feats = [c for c in candidate_features if pd.api.types.is_numeric_dtype(df[c])]
    print("Candidate numeric features:", numeric_feats)

    # remove low-variance / >90% zeros early
    low_var = [c for c in numeric_feats if is_almost_constant(df[c], zero_frac_thresh=0.9)]
    if low_var:
        print("Dropping low-variance / near-constant features:", low_var)
    selected = [c for c in numeric_feats if c not in low_var]

    # leak detection: remove features that perfectly or almost perfectly reproduce label
    leak_removed = []
    for f in selected[:]:
        acc_thresh, thr, direction = find_threshold_predictor(df[f], df[label_col], require_accuracy=0.999)
        if acc_thresh is not None:
            print(f"Detected possible leakage via feature {f} — threshold predictor acc={acc_thresh:.4f}, thr={thr}, dir={direction}. Removing feature.")
            leak_removed.append(f)
            selected.remove(f)
            continue
        # also check very high correlation (monotonic)
        if pd.api.types.is_numeric_dtype(df[f]):
            corr = df[f].corr(df[label_col])
            if abs(corr) > 0.80:
                print(f"High correlation with label: {f} corr={corr:.3f}. Removing feature to avoid leakage.")
                leak_removed.append(f)
                selected.remove(f)

    if not selected:
        raise SystemExit("Depois da filtragem não há features suficientes para treinar.")

    print("Final selected features:", selected)

    # prepare X/y and coerce numeric, impute
    X_train = train_df[selected].copy()
    X_val = val_df[selected].copy()
    y_train = train_df[label_col].astype(int).copy()
    y_val = val_df[label_col].astype(int).copy()

    # coerce types
    for c in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[c]):
            try:
                X_train[c] = pd.to_numeric(X_train[c])
                X_val[c] = pd.to_numeric(X_val[c])
            except Exception:
                print(f"Warning: feature {c} non-numeric and cannot be coerced — dropping.")
                X_train.drop(columns=[c], inplace=True, errors=True)
                X_val.drop(columns=[c], inplace=True, errors=True)
                selected.remove(c)

    if not selected:
        raise SystemExit("Nenhuma feature numérica disponível após coerção.")

    # impute: counts/pct -> 0, others -> median
    for c in selected:
        if c.startswith("num_") or c.startswith("pct_"):
            fill = 0
        else:
            med = X_train[c].median() if not X_train[c].dropna().empty else 0
            fill = med
        X_train[c] = X_train[c].fillna(fill)
        X_val[c] = X_val[c].fillna(fill)

    # scale (Robust)
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=selected, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=selected, index=X_val.index)
    joblib.dump(scaler, scaler_path)
    print("Saved scaler to", scaler_path)

    # final shapes & class balance
    print("X_train shape:", X_train_scaled.shape, "X_val shape:", X_val_scaled.shape)
    print("Label distribution (train):\n", y_train.value_counts())
    print("Label distribution (val):\n", y_val.value_counts())

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = (neg / (pos + 1e-9)) if pos > 0 else 1.0
    print("scale_pos_weight:", scale_pos_weight)

    # LightGBM params: conservative to avoid -inf best gain warnings
    params = dict(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        num_leaves=31,          # moderado; maior -> potencial overfit
        max_depth=6,            # evita árvores muito profundas
        min_child_samples=5,    # evita folhas com poucos exemplos; reduzir para 1 se dataset pequeno
        min_split_gain=0.0,     # permite splits com ganho pequeno (evita -inf)
        min_child_weight=1e-3,  # protege contra divisões numéricas
        reg_alpha=0.0,
        reg_lambda=0.0,
        max_bin=255,            # controle do histograma; às vezes ajuda
        force_row_wise=True,    # evita overhead col-wise e pode reduzir warnings em Windows
        verbosity=-1,
    )

    lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
    lgb_val = lgb.Dataset(X_val_scaled, label=y_val, reference=lgb_train)

    print("LightGBM version:", lgb.__version__)
    bst = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
    )

    # evaluation
    preds_val = bst.predict(X_val_scaled, num_iteration=getattr(bst, "best_iteration", None))
    auc = roc_auc_score(y_val, preds_val) if len(np.unique(y_val)) > 1 else float("nan")
    preds_label = (preds_val >= 0.5).astype(int)
    acc = accuracy_score(y_val, preds_label)
    print(f"\nValidation AUC: {auc:.4f}")
    print(f"Validation Accuracy (0.5 cutoff): {acc:.4f}")
    print("\nClassification report (val):")
    print(classification_report(y_val, preds_label, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_val, preds_label))

    # feature importance
    try:
        fi = pd.DataFrame({"feature": bst.feature_name(), "importance": bst.feature_importance()})
        print("\nFeature importance:\n", fi.sort_values("importance", ascending=False))
    except Exception as e:
        print("Could not extract feature importance:", e)

    # save artifacts
    joblib.dump(bst, model_path)
    with open(features_path, "w", encoding="utf8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print("\nSaved model to:", model_path)
    print("Saved feature list to:", features_path)
    print("Best iteration:", getattr(bst, "best_iteration", None))
