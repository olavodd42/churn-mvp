#!/usr/bin/env python3
"""
scripts/prepare_user_features_v3_penult.py

Gera data/user_features_v3_penult_fixed.parquet com janelas temporais e features derivadas
baseadas no PENÚLTIMO evento (por usuário). Compatível com model/build_training_dataset.py.

Uso:
    python scripts/prepare_user_features_v3_penult.py --events data/events.parquet --out data/user_features_v3_penult_fixed.parquet

Opções importantes:
    --nth-from-last N    -> define qual "as_of" escolher: 1 = último (default), 2 = penúltimo, 3 = antepenúltimo...
                           aqui vamos usar 2 por default para penúltimo.
    --drop-zero-frac / --drop-nan-frac -> thresholds de diagnóstico (não drop automático).

Créditos:
    Adaptado da prepare_user_features_v3.py e estendido para suportar "nth-from-last as_of".
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVENTS = ROOT / "data" / "events.parquet"
DEFAULT_OUT = ROOT / "data" / "user_features_v3_penult_fixed.parquet"
META_PATH = ROOT / "data" / "user_features_v3_penult_meta.json"
ENTITY_SAMPLE_PATH = ROOT / "data" / "entity_sample_penult.parquet"

# windows to compute
WINDOWS = [7, 14, 30, 60, 90]

# defaults for diagnostics
DEFAULT_DROP_ZERO_FRAC = 0.95
DEFAULT_DROP_NAN_FRAC = 0.6
MIN_UNIQUE_FOR_SPLIT = 2

RANDOM_STATE = 42

def read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Arquivo não encontrado: {path}")
    ev = pd.read_parquet(path)
    # normalize names
    if "event_ts" not in ev.columns and "ts" in ev.columns:
        ev = ev.rename(columns={"ts": "event_ts"})
    if "user_id" not in ev.columns:
        raise SystemExit("events.parquet precisa conter 'user_id'")
    ev["event_ts"] = pd.to_datetime(ev["event_ts"], utc=True, errors="coerce")
    if ev["event_ts"].isna().mean() > 0.0:
        mask = ev["event_ts"].isna()
        try:
            ev.loc[mask, "event_ts"] = pd.to_datetime(ev.loc[mask, "event_ts"].astype("int64"), unit="s", utc=True)
        except Exception:
            try:
                ev.loc[mask, "event_ts"] = pd.to_datetime(ev.loc[mask, "event_ts"].astype("int64"), unit="ms", utc=True)
            except Exception:
                pass
    ev["user_id"] = ev["user_id"].astype(int)
    if "is_purchase" not in ev.columns:
        ev["is_purchase"] = (~ev.get("order_value", pd.Series(dtype=float)).isna()).astype(int)
    ev["order_value"] = ev.get("order_value", pd.Series(dtype=float)).astype(float)
    if ev["event_ts"].isna().any():
        n_bad = int(ev["event_ts"].isna().sum())
        print(f"Warning: {n_bad} events have NaT event_ts and will be dropped.")
        ev = ev[ev["event_ts"].notna()].copy()
    # sort for deterministic behavior
    ev = ev.sort_values(["user_id", "event_ts"])
    return ev

def build_nth_from_last_as_of(events: pd.DataFrame, n_from_last: int = 2) -> pd.DataFrame:
    """
    Para cada usuário, escolhe o N-from-last evento como as_of_ts:
      n_from_last=1 -> último evento
      n_from_last=2 -> penúltimo evento (default)
    Se um usuário tem menos que n_from_last eventos, usamos o último evento (com aviso).
    Retorna DataFrame(user_id, as_of_ts)
    """
    # contamos eventos por usuário e pegamos a lista
    grouped = events.groupby("user_id", sort=False)["event_ts"].agg(list).rename("events_list").reset_index()
    as_of_rows = []
    n_users_low = 0
    for r in grouped.itertuples(index=False):
        uid = r.user_id
        elist = r.events_list
        if len(elist) >= n_from_last:
            # pick nth-from-last -> index -n_from_last
            as_of_ts = elist[-n_from_last]
        else:
            # fallback to last event
            as_of_ts = elist[-1]
            n_users_low += 1
        as_of_rows.append({"user_id": int(uid), "as_of_ts": pd.to_datetime(as_of_ts)})
    if n_users_low > 0:
        print(f"Warning: {n_users_low} users have fewer than {n_from_last} events; using last event as fallback for them.")
    out = pd.DataFrame(as_of_rows)
    # ensure tz-aware
    out["as_of_ts"] = pd.to_datetime(out["as_of_ts"], utc=True)
    return out

def vectorized_window_aggregates(events: pd.DataFrame, as_of: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Igual ao V3: faz join events x as_of e calcula janelas.
    """
    merged = events.merge(as_of, on="user_id", how="inner", suffixes=("","_asof"))
    merged = merged[merged["event_ts"] <= merged["as_of_ts"]].copy()
    if merged.empty:
        out = as_of.copy()
        for w in windows:
            out[f"num_orders_{w}d"] = 0
            out[f"sum_amount_{w}d"] = 0.0
            out[f"avg_amount_{w}d"] = np.nan
            out[f"std_amount_{w}d"] = 0.0
        out["n_events_total"] = 0
        out["last_order_ts"] = pd.NaT
        out["recency_days"] = np.nan
        out["pct_active_60d"] = 0.0
        return out

    merged["delta_days"] = (merged["as_of_ts"] - merged["event_ts"]).dt.total_seconds() / 86400.0
    merged["is_purchase"] = merged["is_purchase"].fillna(0).astype(int)
    merged["event_day"] = merged["event_ts"].dt.floor("D")

    agg_frames = []
    for w in windows:
        mask = (merged["delta_days"] > 0.0) & (merged["delta_days"] <= float(w)) & (merged["is_purchase"] == 1)
        dfw = merged.loc[mask, ["user_id", "order_value", "event_day", "event_ts"]].copy()
        if not dfw.empty:
            g = dfw.groupby("user_id", sort=False).agg(
                **{
                    f"num_orders_{w}d": ("event_ts", "count"),
                    f"sum_amount_{w}d": ("order_value", "sum"),
                    f"avg_amount_{w}d": ("order_value", "mean"),
                    f"std_amount_{w}d": ("order_value", "std"),
                    f"active_days_{w}d": ("event_day", lambda s: s.nunique()),
                    f"last_order_{w}d": ("event_ts", "max"),
                }
            )
        else:
            g = pd.DataFrame(index=as_of["user_id"].unique())
            for col in [f"num_orders_{w}d", f"sum_amount_{w}d", f"avg_amount_{w}d", f"std_amount_{w}d", f"active_days_{w}d", f"last_order_{w}d"]:
                g[col] = np.nan
        if f"num_orders_{w}d" in g.columns:
            g[f"num_orders_{w}d"] = g[f"num_orders_{w}d"].fillna(0).astype(int)
        if f"active_days_{w}d" in g.columns:
            g[f"active_days_{w}d"] = g[f"active_days_{w}d"].fillna(0).astype(int)
        for col in [f"sum_amount_{w}d", f"avg_amount_{w}d", f"std_amount_{w}d"]:
            if col in g.columns:
                g[col] = g[col].astype(float)
        g = g.reset_index()
        agg_frames.append(g)

    total_events = merged.groupby("user_id", sort=False).size().rename("n_events_total").reset_index()
    purchases = merged[merged["is_purchase"] == 1].copy()
    last_purchase = purchases.groupby("user_id", sort=False)["event_ts"].max().rename("last_order_ts").reset_index()
    first_purchase = purchases.groupby("user_id", sort=False)["event_ts"].min().rename("first_order_ts").reset_index()

    out = as_of.copy()
    out = out.merge(total_events, on="user_id", how="left")
    out = out.merge(last_purchase, on="user_id", how="left")
    out = out.merge(first_purchase, on="user_id", how="left")
    out["n_events_total"] = out["n_events_total"].fillna(0).astype(int)

    for g in agg_frames:
        out = out.merge(g, on="user_id", how="left")

    out["recency_days"] = (out["as_of_ts"] - out["last_order_ts"]).dt.total_seconds() / 86400.0
    out["recency_days"] = out["recency_days"].where(~out["recency_days"].isna(), other=np.nan)

    if "active_days_60d" in out.columns:
        out["pct_active_60d"] = out["active_days_60d"].fillna(0).astype(int) / 60.0
    else:
        out["pct_active_60d"] = 0.0

    for w in windows:
        stdc = f"std_amount_{w}d"
        if stdc in out.columns:
            out[stdc] = out[stdc].fillna(0.0).astype(float)

    out["num_orders_30d"] = out.get("num_orders_30d", 0).fillna(0).astype(int)
    out["num_orders_60d"] = out.get("num_orders_60d", 0).fillna(0).astype(int)
    out["delta_orders_30_60"] = out["num_orders_30d"] - out["num_orders_60d"]

    out["avg_amount_7d"] = out.get("avg_amount_7d")
    out["avg_amount_30d"] = out.get("avg_amount_30d")
    out["value_slope_7_30"] = (out["avg_amount_7d"] - out["avg_amount_30d"]).astype(float)

    overall_q90 = merged.loc[merged["is_purchase"] == 1, "order_value"].quantile(0.90) if not merged[merged["is_purchase"]==1].empty else np.nan
    out["recent_big_spender"] = ((out["avg_amount_7d"].fillna(0) >= overall_q90) & (out["avg_amount_7d"].notna())).astype(int)
    out["sudden_drop_flag"] = ((out["num_orders_30d"] < 1) & (out["num_orders_60d"] >= 1)).astype(int)

    out["orders_30d"] = out.get("num_orders_30d", 0).astype(float)
    fallback = None
    if "avg_amount_60d" in out.columns:
        fallback = out["avg_amount_60d"]
    elif "avg_amount_30d" in out.columns:
        fallback = out["avg_amount_30d"]
    if fallback is not None:
        out["avg_order_value"] = fallback.fillna(fallback.median()).astype(float)
    else:
        if not merged[merged["is_purchase"]==1].empty:
            global_median = merged.loc[merged["is_purchase"]==1, "order_value"].median()
        else:
            global_median = 0.0
        out["avg_order_value"] = global_median

    out["last_order_ts"] = pd.to_datetime(out["last_order_ts"])
    out["first_order_ts"] = pd.to_datetime(out["first_order_ts"])
    return out

def postprocess_and_buckets(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    for w in windows:
        cnum = f"num_orders_{w}d"
        csum = f"sum_amount_{w}d"
        cavg = f"avg_amount_{w}d"
        cstd = f"std_amount_{w}d"
        if cnum in df.columns:
            df[cnum] = df[cnum].fillna(0).astype(int)
        if csum in df.columns:
            df[csum] = df[csum].fillna(0.0).astype(float)
        if cavg in df.columns:
            df[cavg] = df[cavg].astype(float)
        if cstd in df.columns:
            df[cstd] = df[cstd].fillna(0.0).astype(float)

    df["recency_days_filled"] = df["recency_days"].fillna(9999)
    bins = [-1, 7, 30, 60, 180, 99999]
    labels = ["0-7","8-30","31-60","61-180",">180"]
    df["recency_bucket"] = pd.cut(df["recency_days_filled"], bins=bins, labels=labels)

    df["avg_amount_30d_filled"] = df.get("avg_amount_30d")
    if "avg_amount_60d" in df.columns:
        df["avg_amount_30d_filled"] = df["avg_amount_30d_filled"].fillna(df["avg_amount_60d"])
    global_median = df["avg_amount_30d_filled"].median()
    df["avg_amount_30d_filled"] = df["avg_amount_30d_filled"].fillna(global_median)

    df["freq_ratio_30_60"] = df["num_orders_30d"] / (df["num_orders_60d"].replace(0, np.nan) + 1e-9)
    df["value_ratio_7_60"] = df.get("avg_amount_7d", 0.0) / (df.get("avg_amount_60d", 0.0).replace(0, np.nan) + 1e-9)

    df["log_avg_amount_30d"] = np.log1p(df["avg_amount_30d_filled"].astype(float))

    try:
        df["avg_amount_30d_q"] = pd.qcut(df["avg_amount_30d_filled"], q=5, duplicates="drop").astype(str)
    except Exception:
        df["avg_amount_30d_q"] = pd.cut(df["avg_amount_30d_filled"], bins=5).astype(str)

    keep = [
        "user_id", "as_of_ts", "last_order_ts", "first_order_ts", "recency_days", "recency_bucket",
        "n_events_total", "pct_active_60d",
    ]
    for w in windows:
        keep += [f"num_orders_{w}d", f"sum_amount_{w}d", f"avg_amount_{w}d", f"std_amount_{w}d"]
    keep += [
        "delta_orders_30_60", "value_slope_7_30",
        "recent_big_spender", "sudden_drop_flag",
        "freq_ratio_30_60", "value_ratio_7_60",
        "log_avg_amount_30d", "avg_amount_30d_q",
        "orders_30d", "avg_order_value", "avg_amount_30d_filled"
    ]
    existing_keep = [c for c in keep if c in df.columns]
    return df[existing_keep]

def compute_diagnostics(df: pd.DataFrame, drop_zero_frac: float, drop_nan_frac: float) -> Tuple[List[str], dict]:
    issues = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    high_zero = [c for c in numeric_cols if (df[c]==0).mean() > drop_zero_frac]
    if high_zero:
        issues.append(f"High zero fraction (> {int(drop_zero_frac*100)}%) in numeric features: {high_zero}")
    nan_heavy = [c for c in df.columns if df[c].isna().mean() > drop_nan_frac]
    if nan_heavy:
        issues.append(f"High NaN fraction (> {int(drop_nan_frac*100)}%) in features: {nan_heavy}")
    low_unique = [c for c in numeric_cols if df[c].nunique(dropna=True) < MIN_UNIQUE_FOR_SPLIT]
    if low_unique:
        issues.append(f"Low unique values (<{MIN_UNIQUE_FOR_SPLIT}) in numeric features: {low_unique}")

    meta = {
        "n_rows": int(len(df)),
        "n_users": int(df["user_id"].nunique()) if "user_id" in df.columns else 0,
        "n_numeric": len(numeric_cols),
        "high_zero": high_zero,
        "nan_heavy": nan_heavy,
        "low_unique": low_unique
    }
    return issues, meta

def save_parquet(df: pd.DataFrame, path: Path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression="snappy")
    print(f"Saved {path} ({df.shape[0]} rows, {df.shape[1]} cols)")

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--events", default=str(DEFAULT_EVENTS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--nth-from-last", type=int, default=2,
                   help="which event from the end to use as as_of (2 = penultimate).")
    p.add_argument("--sample-entity", action="store_true")
    p.add_argument("--drop-zero-frac", type=float, default=None,
                   help=f"threshold for high-zero detection (default {DEFAULT_DROP_ZERO_FRAC})")
    p.add_argument("--drop-nan-frac", type=float, default=None,
                   help=f"threshold for high-NaN detection (default {DEFAULT_DROP_NAN_FRAC})")
    args = p.parse_args(argv)

    drop_zero_frac = args.drop_zero_frac if args.drop_zero_frac is not None else DEFAULT_DROP_ZERO_FRAC
    drop_nan_frac = args.drop_nan_frac if args.drop_nan_frac is not None else DEFAULT_DROP_NAN_FRAC

    events = read_events(Path(args.events))
    print("Events loaded:", events.shape)

    as_of = build_nth_from_last_as_of(events, n_from_last=args.nth_from_last)
    print("Users (as_of) built:", as_of.shape)

    features_raw = vectorized_window_aggregates(events, as_of, WINDOWS)
    print("Raw features shape:", features_raw.shape)

    features = postprocess_and_buckets(features_raw, WINDOWS)
    print("Postprocessed shape:", features.shape)

    issues, meta = compute_diagnostics(features, drop_zero_frac, drop_nan_frac)
    if issues:
        print("Quality issues detected:")
        for it in issues:
            print(" -", it)
    else:
        print("Basic quality checks passed.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(features, out_path)

    if args.sample_entity:
        sample = features.sample(min(1000, len(features)), random_state=RANDOM_STATE)[["user_id","as_of_ts"]]
        save_parquet(sample, Path(ENTITY_SAMPLE_PATH))
        print("Saved entity sample to", ENTITY_SAMPLE_PATH)

    meta_out = {
        **meta,
        "windows_days": WINDOWS,
        "created_by": "prepare_user_features_v3_penult.py",
        "drop_zero_frac": float(drop_zero_frac),
        "drop_nan_frac": float(drop_nan_frac),
        "nth_from_last": int(args.nth_from_last)
    }
    Path(META_PATH).write_text(json.dumps(meta_out, indent=2, default=str))
    print("Wrote metadata to", META_PATH)
    print("Done. Output:", args.out)

if __name__ == "__main__":
    main()
