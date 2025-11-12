# scripts/fix_timestamps_and_dtypes.py
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

p = Path("data/user_features.parquet")
if not p.exists():
    raise SystemExit("Arquivo não encontrado: data/user_features.parquet")

df = pd.read_parquet(p)
print("Antes - dtypes:\n", df.dtypes)

# 1) Garantir event_ts como datetime com UTC
if "event_ts" in df.columns:
    if not np.issubdtype(df["event_ts"].dtype, np.datetime64):
        # se for numeric epoch (segundos ou ms) - detecta automaticamente
        sample = df["event_ts"].dropna().iloc[0]
        if isinstance(sample, (int, np.integer)):
            # heurística: se valores grandes (>1e12) então epoch ms, caso contrário epoch s
            if sample > 10**12:
                df["event_ts"] = pd.to_datetime(df["event_ts"], unit="ms", utc=True)
            else:
                df["event_ts"] = pd.to_datetime(df["event_ts"], unit="s", utc=True)
        else:
            df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)
    else:
        # é datetime64[ns] mas possivelmente sem tz -> localiza como UTC
        if df["event_ts"].dt.tz is None:
            df["event_ts"] = df["event_ts"].dt.tz_localize("UTC")
else:
    raise SystemExit("Coluna 'event_ts' não encontrada no parquet - ajuste o FileSource ou gere o parquet corretamente.")

# 2) Garantir last_order_ts como datetime com UTC (se existir)
if "last_order_ts" in df.columns:
    if not np.issubdtype(df["last_order_ts"].dtype, np.datetime64):
        sample = df["last_order_ts"].dropna().iloc[0]
        if isinstance(sample, (int, np.integer)):
            if sample > 10**12:
                df["last_order_ts"] = pd.to_datetime(df["last_order_ts"], unit="ms", utc=True)
            else:
                df["last_order_ts"] = pd.to_datetime(df["last_order_ts"], unit="s", utc=True)
        else:
            df["last_order_ts"] = pd.to_datetime(df["last_order_ts"], utc=True)
    else:
        if df["last_order_ts"].dt.tz is None:
            df["last_order_ts"] = df["last_order_ts"].dt.tz_localize("UTC")

# 3) Forçar dtypes numéricos compatíveis
if "user_id" in df.columns:
    df["user_id"] = df["user_id"].astype("int64")
if "orders_30d" in df.columns:
    df["orders_30d"] = df["orders_30d"].astype("float64")
if "avg_order_value" in df.columns:
    df["avg_order_value"] = df["avg_order_value"].astype("float64")

print("\nDepois - dtypes:\n", df.dtypes)
print(df.head(3))

# 4) Reescrever parquet com pyarrow preservando timezone
table = pa.Table.from_pandas(df)
pq.write_table(table, "data/user_features.parquet", compression="snappy")
print("\nReescreveu data/user_features.parquet com timestamps timezone-aware (UTC).")
