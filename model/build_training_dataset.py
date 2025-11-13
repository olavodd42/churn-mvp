#!/usr/bin/env python3
"""
model/build_training_dataset.py  (robust + compatible)

Gera:
 - data/training_dataset.parquet
 - data/training_dataset_penult.parquet  (se user_feat é penult)
 - model/artifacts/feature_list_v2.json
 - data/training_dataset_meta.json

Comportamento:
 - procura automaticamente por `data/user_features_v3_penult.parquet`;
   se não existir, usa `data/user_features_v3.parquet`.
 - calcula next_purchase (min event_ts > as_of_ts) e days_to_next
 - label = 1 (churn) se next_purchase is NaT OR days_to_next > churn_window_days
 - label = 0 se next_purchase exists e days_to_next <= churn_window_days
 - seleciona features numéricas robustamente com fallback.

Use:
    python model/build_training_dataset.py
    python model/build_training_dataset.py --user-feat path/to/user_features.parquet
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PENULT = ROOT / "data" / "user_features_v3_penult.parquet"
DEFAULT_USER_FEAT = ROOT / "data" / "user_features_v3.parquet"
EVENTS = ROOT / "data" / "events.parquet"
OUT = ROOT / "data" / "training_dataset.parquet"
OUT_PENULT = ROOT / "data" / "training_dataset_penult.parquet"
META_OUT = ROOT / "data" / "training_dataset_meta.json"
FEATURE_LIST = ROOT / "model" / "artifacts" / "feature_list_v2.json"

CHURN_WINDOW_DAYS = 30

def choose_user_feat_default() -> Path:
    """Prefer penult features if available for safe labeling (no leakage)."""
    if DEFAULT_PENULT.exists():
        return DEFAULT_PENULT
    if DEFAULT_USER_FEAT.exists():
        return DEFAULT_USER_FEAT
    raise SystemExit("Nenhum arquivo user_features encontrado. Gere com scripts/prepare_user_features_v3.py")

def load_user_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"user features não encontrado: {path}")
    df = pd.read_parquet(path)
    if "as_of_ts" not in df.columns:
        raise SystemExit("user_features não contém coluna 'as_of_ts' — verifique o script de features.")
    df["as_of_ts"] = pd.to_datetime(df["as_of_ts"], utc=True, errors="coerce")
    if df["as_of_ts"].isna().any():
        n = int(df["as_of_ts"].isna().sum())
        print(f"Warning: {n} linhas com as_of_ts NaT — serão descartadas.")
        df = df[df["as_of_ts"].notna()].copy()
    return df

def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"events não encontrado: {path}")
    ev = pd.read_parquet(path)
    if "event_ts" not in ev.columns and "ts" in ev.columns:
        ev = ev.rename(columns={"ts": "event_ts"})
    if "event_ts" not in ev.columns:
        raise SystemExit("events.parquet não contém coluna 'event_ts' nem 'ts'.")
    ev["event_ts"] = pd.to_datetime(ev["event_ts"], utc=True, errors="coerce")
    if ev["event_ts"].isna().any():
        n = int(ev["event_ts"].isna().sum())
        print(f"Warning: {n} events have NaT event_ts -> dropping those rows")
        ev = ev[ev["event_ts"].notna()].copy()
    if "user_id" not in ev.columns:
        raise SystemExit("events.parquet precisa conter 'user_id'")
    ev["user_id"] = ev["user_id"].astype(int)
    if "is_purchase" not in ev.columns:
        ev["is_purchase"] = (~ev.get("order_value", pd.Series(dtype=float)).isna()).astype(int)
    ev["is_purchase"] = ev["is_purchase"].astype(int)
    return ev

def compute_next_purchase(events: pd.DataFrame, user_feats: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada user (as_of_ts) encontra next_purchase_ts (min event_ts > as_of_ts).
    Retorna DataFrame user_id, as_of_ts, next_purchase_ts, days_to_next (float days).
    """
    purchases = events[events["is_purchase"] == 1].copy()
    # join user as_of to purchases on user_id: keep all users (right join)
    merged = user_feats[["user_id","as_of_ts"]].merge(
        purchases[["user_id","event_ts"]], on="user_id", how="left"
    )
    # keep only purchases after as_of_ts
    mask_after = (merged["event_ts"].notna()) & (merged["event_ts"] > merged["as_of_ts"])
    merged_after = merged[mask_after].copy()
    if merged_after.empty:
        out = user_feats[["user_id","as_of_ts"]].copy()
        out["next_purchase_ts"] = pd.NaT
        out["days_to_next"] = np.nan
        return out
    next_per_user = merged_after.groupby("user_id", sort=False)["event_ts"].min().rename("next_purchase_ts").reset_index()
    out = user_feats[["user_id","as_of_ts"]].merge(next_per_user, on="user_id", how="left")
    out["days_to_next"] = (out["next_purchase_ts"] - out["as_of_ts"]).dt.total_seconds() / 86400.0
    return out

def robust_choose_features(df: pd.DataFrame) -> List[str]:
    """
    Escolha robusta de features:
     - exclui colunas óbvias (ids/timestamps/label/next_purchase_ts)
     - seleciona numéricas com >1 unique
     - se nada for selecionado, tenta um fallback por prefixos conhecidos
    """
    exclude = {"user_id","as_of_ts","last_order_ts","first_order_ts","next_purchase_ts","label"}
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c].dtype)]
    keep = [c for c in sorted(numeric_cols) if df[c].nunique(dropna=True) > 1]

    if keep:
        return keep

    # fallback: try to include engineered features by prefix
    prefixes = ["num_orders_","avg_amount_","sum_amount_","std_amount_","orders_","pct_active_","recency_","log_avg_amount","value_ratio","delta_orders"]
    fallback = []
    for p in prefixes:
        fallback += [c for c in df.columns if c.startswith(p) and c not in exclude and pd.api.types.is_numeric_dtype(df[c].dtype)]
    fallback = sorted(list(dict.fromkeys(fallback)))  # unique preserving order
    fallback = [c for c in fallback if df[c].nunique(dropna=True) > 1]
    if fallback:
        print("Warning: nenhum feature numérico padrão encontrado; usando fallback por prefixos.")
        return fallback

    # last resort: include any non-constant numeric column (even if 1 unique it was filtered; double-check)
    any_numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dtype) and c not in exclude]
    any_numeric = [c for c in any_numeric if df[c].nunique(dropna=True) > 0]
    if any_numeric:
        print("Warning: usando último recurso - incluindo any_numeric cols.")
        return sorted(any_numeric)

    # nothing to do
    return []

def ensure_no_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If duplicate column names exist, rename duplicates by appending suffix _dupN.
    Returns a dataframe with unique column names.
    """
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_name = f"{c}_dup{seen[c]}"
            # avoid colliding new_name with existing names
            while new_name in seen:
                seen[c] += 1
                new_name = f"{c}_dup{seen[c]}"
            seen[new_name] = 0
            new_cols.append(new_name)
    df.columns = new_cols
    return df

def save_parquet(df: pd.DataFrame, path: Path):
    # ensure unique columns for pyarrow
    df = ensure_no_duplicate_columns(df)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression="snappy")
    print(f"Saved {path} ({df.shape[0]} rows, {df.shape[1]} cols)")

def main(user_feat_path: str = None):
    # decide user features path
    if user_feat_path:
        USER_FEAT = Path(user_feat_path)
    else:
        USER_FEAT = choose_user_feat_default()

    print("Loading user features from:", USER_FEAT)
    user_feats = load_user_features(USER_FEAT)

    print("Loading events...")
    events = load_events(EVENTS)

    print("Computing next purchase per user (vectorized)...")
    next_df = compute_next_purchase(events, user_feats)
    n_no_next = int(next_df["next_purchase_ts"].isna().sum())
    print(f"Users without next purchase after as_of: {n_no_next} / {len(next_df)}")

    print(f"Building labels with churn window = {CHURN_WINDOW_DAYS} days...")
    ds = user_feats.merge(next_df[["user_id","next_purchase_ts","days_to_next"]], on="user_id", how="left")

    ds["label"] = ((ds["days_to_next"].isna()) | (ds["days_to_next"] > float(CHURN_WINDOW_DAYS))).astype(int)

    pos = int(ds["label"].sum())
    neg = int((ds["label"]==0).sum())
    print(f"Dataset rows: {len(ds)}  positives: {pos} negatives: {neg}")

    if pos == 0 or neg == 0:
        print("WARNING: all labels identical — cheque a lógica de labeling ou seus eventos (possível falta de purchases after as_of).")
        print("Sample head:", ds.head(10).to_dict(orient='records'))

    features = robust_choose_features(ds)
    if not features:
        raise SystemExit("Nenhuma feature numérica útil detectada. Verifique user_features_v3 (ou o fallback de prefixos).")

    print("Using features:", features)

    # assemble final frame (include debug cols next_purchase_ts/days_to_next)
    final_cols = ["user_id","as_of_ts"] + features + ["next_purchase_ts","days_to_next","label"]
    final = ds[final_cols].copy()
    final["label"] = final["label"].astype(int)

    # Save main OUT
    out_path = OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(final, out_path)

    # Save also penult-named output for compatibility (if input was penult or explicit param)
    try:
        if USER_FEAT.name.find("penult") >= 0 or DEFAULT_PENULT.exists():
            out_penult = OUT_PENULT
            out_penult.parent.mkdir(parents=True, exist_ok=True)
            save_parquet(final, out_penult)
    except Exception:
        # ignore secondary save errors (we already saved primary)
        pass

    # Save feature list (without debug cols)
    FEATURE_LIST.parent.mkdir(parents=True, exist_ok=True)
    json.dump(features, open(FEATURE_LIST, "w"), indent=2)
    print("Saved feature list to", FEATURE_LIST)

    # metadata
    meta = {
        "n_rows": int(len(final)),
        "n_features": len(features),
        "n_positive": pos,
        "n_negative": neg,
        "churn_window_days": CHURN_WINDOW_DAYS,
        "created_by": "build_training_dataset.py",
        "user_feat_source": str(USER_FEAT)
    }
    Path(META_OUT).write_text(json.dumps(meta, indent=2, default=str))
    print("Saved meta to", META_OUT)
    print("Done. Dataset ready for model/train_baseline_v2.py")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--user-feat", type=str, default=None, help="Path to user features parquet (penult preferred).")
    args = p.parse_args()
    main(args.user_feat)
