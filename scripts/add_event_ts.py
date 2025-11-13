#!/usr/bin/env python3
"""
scripts/prepare_user_features_v3.py

Gera data/user_features_v3.parquet com janelas temporais e features derivadas.
- Usa como "as_of" para cada usuário o timestamp do último evento do usuário (garante
  que não usamos dados futuros).
- Cria janelas 7/14/30/60/90 dias, recência, tendências, variabilidade, buckets e flags.
- Salva parquet (pyarrow) preservando timezone UTC.

Entrada esperada (exemplo):
- data/events.parquet com colunas mínimas:
    user_id (int), event_ts (datetime OR epoch), event_type (str) or is_purchase (0/1),
    order_value (float) — order_value pode ser NaN para eventos não purchase.

Saídas:
- data/user_features_v3.parquet
- (opcional) data/entity_sample.parquet com algumas linhas entity_df para debug
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import json
import sys
from typing import List

ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = ROOT / "data" / "events.parquet"
OUT_PATH = ROOT / "data" / "user_features_v3.parquet"
ENTITY_SAMPLE_PATH = ROOT / "data" / "entity_sample.parquet"

WINDOWS = [7, 14, 30, 60, 90]
RANDOM_STATE = 42

def read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Arquivo de eventos não encontrado: {path}")
    df = pd.read_parquet(path)
    # normaliza nomes
    if "event_ts" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "event_ts"})
    if "user_id" not in df.columns:
        raise SystemExit("Coluna 'user_id' não encontrada em events.parquet")
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True, errors="coerce")
    if df["event_ts"].isna().any():
        # tenta heurística para epoch int
        sample = df["event_ts"].isna().sum()
        print(f"Warning: {sample} timestamps não parseáveis; verifique formato. Tentando coerção por epoch...")
        try:
            df.loc[df["event_ts"].isna(), "event_ts"] = pd.to_datetime(
                df.loc[df["event_ts"].isna(), "event_ts"], unit="s", utc=True
            )
        except Exception:
            pass
    return df

def build_user_as_of(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula timestamp 'as_of' por usuário (último evento do usuário)."""
    last = df.groupby("user_id")["event_ts"].max().rename("as_of_ts").reset_index()
    return last

def agg_windows_for_user(df: pd.DataFrame, users_as_of: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Computes windowed aggregates for each user at their as_of_ts.
       Efficient approach: for each window, filter events with event_ts > as_of - window and aggregate.
       If dataset huge, replace with SQL / optimized groupby on partitioned data.
    """
    rows = []
    # prepare order indicator and amount
    if "is_purchase" not in df.columns:
        # infer purchase by presence of order_value non-null
        df["is_purchase"] = (~df.get("order_value", pd.Series(dtype=float)).isna()).astype(int)
    df["order_value"] = df.get("order_value", pd.Series(dtype=float)).astype(float)

    # index events by user for faster per-user slicing (dictionary of frames)
    groups = {uid: sub.sort_values("event_ts") for uid, sub in df.groupby("user_id")}
    total = len(users_as_of)
    for i, row in enumerate(users_as_of.itertuples(index=False), 1):
        uid = row.user_id
        as_of = row.as_of_ts
        user_events = groups.get(uid)
        # default baseline
        feat = {
            "user_id": int(uid),
            "as_of_ts": as_of,
            "last_order_ts": np.nan,
            "recency_days": np.nan,
            "first_order_ts": pd.NaT,
            "n_events_total": 0,
        }
        if user_events is None or user_events.empty:
            # no events for user — keep defaults
            rows.append(feat)
            continue

        # last purchase timestamp (<= as_of)
        purchases = user_events[user_events["is_purchase"] == 1]
        purchases = purchases[purchases["event_ts"] <= as_of]
        if not purchases.empty:
            feat["last_order_ts"] = purchases["event_ts"].max()
            feat["recency_days"] = (as_of - feat["last_order_ts"]).days
            feat["first_order_ts"] = purchases["event_ts"].min()
        else:
            # if no purchase ever, last_order_ts stays NaN; recency large sentinel
            feat["last_order_ts"] = pd.NaT
            feat["recency_days"] = np.nan

        feat["n_events_total"] = int(user_events[user_events["event_ts"] <= as_of].shape[0])

        # window aggregates
        for w in windows:
            start = as_of - pd.Timedelta(days=w)
            window_events = user_events[(user_events["event_ts"] > start) & (user_events["event_ts"] <= as_of)]
            purchases_w = window_events[window_events["is_purchase"] == 1]
            num_orders = purchases_w.shape[0]
            sum_amount = float(purchases_w["order_value"].sum()) if num_orders > 0 else 0.0
            avg_amount = float(purchases_w["order_value"].mean()) if num_orders > 0 else np.nan
            std_amount = float(purchases_w["order_value"].std()) if num_orders > 1 else 0.0
            # store
            feat[f"num_orders_{w}d"] = int(num_orders)
            feat[f"sum_amount_{w}d"] = sum_amount
            feat[f"avg_amount_{w}d"] = avg_amount
            feat[f"std_amount_{w}d"] = std_amount

        # pct_active_60d: fraction of days with at least one purchase in last 60 days
        w60_start = as_of - pd.Timedelta(days=60)
        ev60 = purchases[(purchases["event_ts"] > w60_start) & (purchases["event_ts"] <= as_of)]
        if ev60.empty:
            feat["pct_active_60d"] = 0.0
        else:
            active_days = ev60["event_ts"].dt.floor("D").nunique()
            feat["pct_active_60d"] = float(active_days / 60.0)

        # variability across windows sample: simple slope order counts
        # delta orders 30 vs 60
        n30 = feat.get("num_orders_30d", 0)
        n60 = feat.get("num_orders_60d", 0)
        feat["delta_orders_30_60"] = int(n30 - n60)
        # value slope
        a7 = feat.get("avg_amount_7d", np.nan)
        a30 = feat.get("avg_amount_30d", np.nan)
        feat["value_slope_7_30"] = float(a7 - a30) if not (np.isnan(a7) or np.isnan(a30)) else np.nan

        # flags
        feat["recent_big_spender"] = int(a7 > 0 and not np.isnan(a7) and a7 >= df["order_value"].quantile(0.90))
        feat["sudden_drop_flag"] = int((n30 < 1) and (n60 >= 1))

        rows.append(feat)

        # progress small print for big datasets
        if i % 5000 == 0:
            print(f"Processed {i}/{total} users...")

    out = pd.DataFrame(rows)
    return out

def postprocess_and_buckets(df: pd.DataFrame) -> pd.DataFrame:
    # fill na for numeric aggregates with sensible defaults
    num_cols = [c for c in df.columns if c.startswith(("num_orders_","sum_amount_","avg_amount_","std_amount_"))]
    for c in num_cols:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(0)

    # recency: fill NaN (never ordered) with large sentinel + clip
    df["recency_days"] = df["recency_days"].fillna(999).astype(int)
    # bucket recency
    bins = [-1, 7, 30, 60, 180, 9999]
    labels = ["0-7","8-30","31-60","61-180",">180"]
    df["recency_bucket"] = pd.cut(df["recency_days"], bins=bins, labels=labels)

    # avg value fallback: if avg_amount_30d NaN, fallback to avg_amount_60d then global avg
    df["avg_amount_30d_filled"] = df["avg_amount_30d"].fillna(df["avg_amount_60d"])
    global_median = df["avg_amount_30d_filled"].median()
    df["avg_amount_30d_filled"] = df["avg_amount_30d_filled"].fillna(global_median)

    # ratio features (safely)
    df["freq_ratio_30_60"] = df["num_orders_30d"] / (df["num_orders_60d"].replace(0, np.nan) + 1e-9)
    df["value_ratio_7_60"] = df["avg_amount_7d"] / (df["avg_amount_60d"].replace(0, np.nan) + 1e-9)

    # clips & transforms
    df["pct_active_60d"] = df["pct_active_60d"].clip(0,1)
    df["log_avg_amount_30d"] = np.log1p(df["avg_amount_30d_filled"])

    # categorical quantile bucket for avg_amount_30d_filled (5 quantiles)
    try:
        df["avg_amount_30d_q"] = pd.qcut(df["avg_amount_30d_filled"], q=5, duplicates="drop").astype(str)
    except Exception:
        # fallback to manual bins
        df["avg_amount_30d_q"] = pd.cut(df["avg_amount_30d_filled"], bins=5).astype(str)

    # keep deterministic column ordering
    keep = [
        "user_id", "as_of_ts", "last_order_ts", "recency_days", "recency_bucket",
        "n_events_total", "pct_active_60d",
    ]
    for w in WINDOWS:
        keep += [f"num_orders_{w}d", f"sum_amount_{w}d", f"avg_amount_{w}d", f"std_amount_{w}d"]
    keep += [
        "delta_orders_30_60", "value_slope_7_30",
        "recent_big_spender", "sudden_drop_flag",
        "freq_ratio_30_60", "value_ratio_7_60",
        "log_avg_amount_30d", "avg_amount_30d_q"
    ]
    existing_keep = [c for c in keep if c in df.columns]
    return df[existing_keep]

def basic_quality_checks(df: pd.DataFrame):
    issues = []
    # no future features: last_order_ts must be <= as_of_ts if present
    if "last_order_ts" in df.columns:
        bad = df[(~df["last_order_ts"].isna()) & (df["last_order_ts"] > df["as_of_ts"])]
        if not bad.empty:
            issues.append(f"Found {len(bad)} rows where last_order_ts > as_of_ts (possible leakage).")
    # sparsity
    sparsity = {c: float((df[c]==0).mean()) for c in df.select_dtypes(include=[np.number]).columns}
    high_zero = [c for c, frac in sparsity.items() if frac > 0.9]
    if high_zero:
        issues.append(f"High zero fraction (>90%) in numeric features: {high_zero}")
    # nans
    nan_heavy = [c for c in df.columns if df[c].isna().mean() > 0.5]
    if nan_heavy:
        issues.append(f"High NaN fraction (>50%) in features: {nan_heavy}")

    return issues

def save_parquet(df: pd.DataFrame, path: Path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression="snappy")
    print(f"Saved {path} ({df.shape[0]} rows, {df.shape[1]} cols)")

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--events", default=str(EVENTS_PATH))
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--sample-entity", action="store_true",
                   help="salvar amostra entity_sample.parquet para debug")
    args = p.parse_args(argv)

    events = read_events(Path(args.events))
    print("Events loaded:", events.shape)

    users_as_of = build_user_as_of(events)
    print("Unique users:", users_as_of.shape[0])

    features = agg_windows_for_user(events, users_as_of, WINDOWS)
    print("Raw features built:", features.shape)

    features = postprocess_and_buckets(features)
    print("Postprocessed features:", features.shape)

    issues = basic_quality_checks(features)
    if issues:
        print("Quality issues detected:")
        for it in issues:
            print(" -", it)
    else:
        print("Basic quality checks passed.")

    out_path = Path(args.out)
    save_parquet(features, out_path)

    if args.sample_entity:
        # grava algumas linhas para debug / entity_df offline
        sample = features.sample(min(1000, len(features)), random_state=RANDOM_STATE)[["user_id","as_of_ts"]]
        save_parquet(sample, Path(ENTITY_SAMPLE_PATH))
        print("Saved entity sample to", ENTITY_SAMPLE_PATH)

    # also dump a small metadata file (useful)
    meta = {
        "n_rows": int(len(features)),
        "n_users": int(features["user_id"].nunique()),
        "windows_days": WINDOWS,
        "created_by": "prepare_user_features_v3.py",
    }
    (out_path.parent / "user_features_v3_meta.json").write_text(json.dumps(meta, indent=2))
    print("Wrote metadata.")

if __name__ == "__main__":
    main()
