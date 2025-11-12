# scripts/debug_model.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score

repo = Path(".").resolve()
data_path = repo / "data" / "training_dataset.parquet"
model_path = repo / "model" / "artifacts" / "churn_baseline_v2.pkl"

df = pd.read_parquet(data_path)
print("Dataset shape:", df.shape)
print(df["label"].value_counts(dropna=False))
print()

# 1) Verificar colunas constantes / NaN fraction
for c in df.columns:
    nunq = df[c].nunique(dropna=False)
    nan_frac = df[c].isna().mean()
    if nunq == 1:
        print(f"CONSTANT column: {c}  (unique={nunq}, nan_frac={nan_frac:.2f})")
    if nan_frac > 0.9:
        print(f"LOTS OF NANs (>90%): {c}")

print("\n--- Equality check: feature == label (exact) ---")
for c in df.columns:
    if c == "label": continue
    # equality after filling NaN with -999 for safe compare
    try:
        eq = (df[c].fillna(-999) == df["label"].fillna(-999)).all()
        if eq:
            print(f"FEATURE IDENTICAL TO LABEL: {c}")
    except Exception:
        pass

print("\n--- High correlation / monotonic relation with label ---")
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "label"]
corrs = []
for c in numeric_cols:
    try:
        corr = df[c].fillna(0).corr(df["label"].fillna(0))
        corrs.append((c, corr))
    except Exception:
        continue
corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
for c, corr in corrs[:20]:
    print(f"{c}: corr with label = {corr:.4f}")

# 2) Check predicted probabilities distribution (if model exists)
if model_path.exists():
    bst = joblib.load(model_path)
    # choose a small feature set intersection
    feat_cols = [c for c in numeric_cols if c in getattr(bst, "feature_name", lambda: [])()]
    if not feat_cols:
        # fallback to numeric cols
        feat_cols = numeric_cols
    X_val = df[feat_cols].fillna(0).astype(float)
    try:
        probs = bst.predict(X_val, num_iteration=getattr(bst, "best_iteration", None))
        print("\nPred probs stats: min, 25%, 50%, 75%, max:")
        print(np.percentile(probs, [0,25,50,75,100]))
        try:
            auc = roc_auc_score(df["label"], probs)
            print("ROC AUC (full):", auc)
        except Exception as e:
            print("Could not compute AUC:", e)
    except Exception as e:
        print("Modelo existe mas não consegui rodar predict:", e)
else:
    print("Modelo não encontrado em", model_path)
