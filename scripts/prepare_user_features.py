# scripts/prepare_user_features.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# lê eventos brutos
df = pd.read_parquet("data/events.parquet")

# garante datetime
df["event_ts"] = pd.to_datetime(df["event_ts"])

# agregação por usuário (exemplo: último evento, soma compras 30d, média valor)
agg = (
    df.sort_values("event_ts")
      .groupby("user_id")
      .agg(
        orders_30d=("is_purchase", "sum"),
        avg_order_value=("order_value", "mean"),
        last_order_ts=("event_ts", "max"),   # nome exato esperado pelo FeatureView
      )
      .reset_index()
)

# salva preservando timestamp
table = pa.Table.from_pandas(agg)
pq.write_table(table, "data/user_features.parquet", compression="snappy")

print("Saved data/user_features.parquet with schema:")
print(table.schema)
print(agg.head())
