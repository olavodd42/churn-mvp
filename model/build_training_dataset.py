#!/usr/bin/env python3
"""
model/build_training_dataset.py  (corrigido + debug-friendly)

Gera data/training_dataset.parquet incluindo as colunas:
 - next_purchase_ts
 - days_to_next

Rótulo temporal:
 - label = 1 (churn) se next_purchase is NaT OR days_to_next > churn_window_days
 - label = 0 (not churn) se next_purchase existe e days_to_next <= churn_window_days
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List

ROOT = Path(__file__).resolve().parent.parent
USER_FEAT = ROOT / "data" / "user_features_v3.parquet"
EVENTS = ROOT / "data" / "events.parquet"
OUT = ROOT / "data" / "training_dataset.parquet"
META_OUT = ROOT / "data" / "training_dataset_meta.json"
FEATURE_LIST = ROOT / "model" / "artifacts" / "feature_list_v2.json"

CHURN_WINDOW_DAYS = 30

def load_user_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"user features não encontrado: {path}")
    df = pd.read_parquet(path)
    if "as_of_ts" not in df.columns:
        raise SystemExit("user_features_v3 não tem coluna 'as_of_ts'")
    df["as_of_ts"] = pd.to_datetime(df["as_of_ts"], utc=True, errors="coerce")
    if df["as_of_ts"].isna().any():
        n = int(df["as_of_ts"].isna().sum())
        print(f"Warning: {n} rows with NaT as_of_ts (dropping)")
        df = df[df["as_of_ts"].notna()].copy()
    return df

def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"events não encontrado: {path}")
    ev = pd.read_parquet(path)
    # normalize ts column names
    if "event_ts" not in ev.columns and "ts" in ev.columns:
        ev = ev.rename(columns={"ts": "event_ts"})
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
    Para cada user (as_of_ts) encontra the next purchase ts (min event_ts > as_of_ts).
    Retorna DataFrame user_id, as_of_ts, next_purchase_ts, days_to_next (float days, NaN if none).
    """
    purchases = events[events["is_purchase"] == 1].copy()
    # join user as_of to purchases on user_id: keep all users (right join)
    merged = user_feats[["user_id","as_of_ts"]].merge(
        purchases[["user_id","event_ts"]], on="user_id", how="left", suffixes=("","_purchase")
    )
    # keep only purchases after as_of_ts
    mask_after = (merged["event_ts"].notna()) & (merged["event_ts"] > merged["as_of_ts"])
    merged_after = merged[mask_after].copy()
    if merged_after.empty:
        out = user_feats[["user_id","as_of_ts"]].copy()
        out["next_purchase_ts"] = pd.NaT
        out["days_to_next"] = np.nan
        return out
    # compute min event_ts per user
    next_per_user = merged_after.groupby("user_id", sort=False)["event_ts"].min().rename("next_purchase_ts").reset_index()
    out = user_feats[["user_id","as_of_ts"]].merge(next_per_user, on="user_id", how="left")
    out["days_to_next"] = (out["next_purchase_ts"] - out["as_of_ts"]).dt.total_seconds() / 86400.0
    return out

def build_training_dataset(user_feats: pd.DataFrame, next_df: pd.DataFrame, churn_window_days: int) -> pd.DataFrame:
    df = user_feats.merge(next_df[["user_id","next_purchase_ts","days_to_next"]], on="user_id", how="left")
    # label: churn=1 if no next purchase in window
    df["label"] = ((df["days_to_next"].isna()) | (df["days_to_next"] > float(churn_window_days))).astype(int)
    return df

def choose_features_for_model(df: pd.DataFrame) -> List[str]:
    # Explicitly exclude debug cols so they don't get into the features list
    exclude = {
        "user_id",
        "as_of_ts",
        "last_order_ts",
        "first_order_ts",
        "next_purchase_ts",
        "days_to_next",
        "label",
    }
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c].dtype)]
    # keep only non-constant
    keep = [c for c in sorted(cols) if df[c].nunique(dropna=True) > 1]
    return keep

def save_parquet(df: pd.DataFrame, path: Path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression="snappy")
    print(f"Saved {path} ({df.shape[0]} rows, {df.shape[1]} cols)")

def main():
    print("Loading user features...")
    user_feats = load_user_features(USER_FEAT)
    print("Loading events...")
    events = load_events(EVENTS)

    print("Computing next purchase per user (vectorized)...")
    next_df = compute_next_purchase(events, user_feats)
    n_no_next = int(next_df["next_purchase_ts"].isna().sum())
    print(f"Users without next purchase after as_of: {n_no_next} / {len(next_df)}")

    print(f"Building labels with churn window = {CHURN_WINDOW_DAYS} days...")
    ds = build_training_dataset(user_feats, next_df, CHURN_WINDOW_DAYS)

    pos = int(ds["label"].sum())
    neg = int((ds["label"]==0).sum())
    print(f"Dataset rows: {len(ds)}  positives: {pos} negatives: {neg}")

    if pos == 0 or neg == 0:
        print("WARNING: all labels identical — cheque a lógica de labeling ou seus eventos (possível falta de purchases after as_of).")
        print("Sample head:", ds.head(10).to_dict(orient='records'))

    # choose features
    features = choose_features_for_model(ds)
    if not features:
        raise SystemExit("Nenhuma feature numérica útil detectada. Verifique user_features_v3.")
    print("Using features:", features)

    # FINAL: include next_purchase_ts and days_to_next for debugging / validation
    # ensure we don't duplicate columns (defensive)
    debug_cols = ["next_purchase_ts", "days_to_next", "label"]
    final_cols = ["user_id","as_of_ts"] + [f for f in features if f not in debug_cols] + debug_cols
    # defensive: preserve order and remove duplicates if any
    seen = set()
    final_cols_unique = []
    for c in final_cols:
        if c not in seen:
            final_cols_unique.append(c)
            seen.add(c)

    final = ds[final_cols_unique].copy()
    final["label"] = final["label"].astype(int)

    # save final dataset
    out_path = OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(final, out_path)

    # save feature list (without debug cols)
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
        "created_by": "build_training_dataset.py"
    }
    Path(META_OUT).write_text(json.dumps(meta, indent=2, default=str))
    print("Saved meta to", META_OUT)
    print("Done. Dataset ready for model/train_baseline_v2.py")

if __name__ == "__main__":
    main()
