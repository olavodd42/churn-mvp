# scripts/add_event_ts_from_last_order.py
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
print(df.head(3))

# Se existir last_order_ts como int -> converte para datetime UTC
if "last_order_ts" in df.columns:
    if not np.issubdtype(df["last_order_ts"].dtype, np.datetime64):
        sample = df["last_order_ts"].dropna().iloc[0] if not df["last_order_ts"].dropna().empty else None
        if isinstance(sample, (int, np.integer)):
            # heurística: epoch ms vs s
            if sample > 10**12:
                df["last_order_ts"] = pd.to_datetime(df["last_order_ts"], unit="ms", utc=True)
            else:
                df["last_order_ts"] = pd.to_datetime(df["last_order_ts"], unit="s", utc=True)
        else:
            df["last_order_ts"] = pd.to_datetime(df["last_order_ts"], utc=True)
# Se não houver event_ts, cria-a igual a last_order_ts (timestamp do último evento)
if "event_ts" not in df.columns:
    if "last_order_ts" in df.columns:
        df["event_ts"] = df["last_order_ts"]
    else:
        raise SystemExit("Nenhuma coluna de timestamp encontrada (last_order_ts). Gere o parquet corretamente.")

# Forçar dtypes numéricos compatíveis
if "user_id" in df.columns:
    df["user_id"] = df["user_id"].astype("int64")
if "orders_30d" in df.columns:
    df["orders_30d"] = df["orders_30d"].astype("float64")
if "avg_order_value" in df.columns:
    df["avg_order_value"] = df["avg_order_value"].astype("float64")

print("\nDepois - dtypes:\n", df.dtypes)
print(df.head(3))

# Reescreve parquet preservando timestamps com pyarrow
table = pa.Table.from_pandas(df)
pq.write_table(table, "data/user_features.parquet", compression="snappy")
print("\nReescreveu data/user_features.parquet (event_ts adicionado e timestamps UTC).")
