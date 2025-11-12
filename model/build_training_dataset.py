# model/build_training_dataset.py
"""
Gera o dataset de treino com features derivadas (recência, frequência, média de valor, pct dias ativos)
e combina com features do Feast. Sem vazamento: todas as janelas usam eventos estritamente
anteriores ao label timestamp (event_timestamp).

Rode:
    .venv\Scripts\Activate
    python model\build_training_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from feast import FeatureStore

# ------------------ PATHS ------------------
repo_root = Path(__file__).resolve().parent.parent
data_dir = repo_root / "data"
user_features_path = data_dir / "user_features.parquet"
events_path = data_dir / "events.parquet"              # eventos brutos recomendados
output_path = data_dir / "training_dataset.parquet"
feature_repo_path = repo_root / "feature_repo"

# ------- checagem de arquivos mínimos -------
if not user_features_path.exists():
    raise SystemExit(f"Arquivo não encontrado: {user_features_path}. Gere com scripts/prepare_user_features.py")

# carrega snapshot / agregados por usuário (fallback)
df_users = pd.read_parquet(user_features_path)
print("Loaded user_features:", user_features_path, "shape:", df_users.shape)
print("Columns:", df_users.columns.tolist())

# escolhe coluna de timestamp no snapshot
ts_col = "event_ts" if "event_ts" in df_users.columns else ("last_order_ts" if "last_order_ts" in df_users.columns else None)
if ts_col is None:
    raise SystemExit("Nenhuma coluna de timestamp encontrada em user_features.parquet (procure por 'event_ts' ou 'last_order_ts').")
print("Using snapshot timestamp column:", ts_col)

# tenta carregar events.parquet (fonte ideal para janelas)
use_events = False
if events_path.exists():
    events = pd.read_parquet(events_path)
    # normaliza nomes
    if "event_ts" not in events.columns and "timestamp" in events.columns:
        events = events.rename(columns={"timestamp": "event_ts"})
    if "event_ts" not in events.columns:
        raise SystemExit("events.parquet encontrado mas não contém coluna 'event_ts' nem 'timestamp'.")
    events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True)
    # garantir colunas mínimas (user_id, event_ts). order_value and is_purchase são opcionais mas usados se existirem.
    if "user_id" not in events.columns:
        raise SystemExit("events.parquet precisa conter coluna 'user_id'.")
    print("Using events.parquet for windowed aggregations. Events shape:", events.shape)
    use_events = True
else:
    events = None
    print("events.parquet not found — falling back to snapshot history for approximations.")

# ------------------ MONTAR entity_df ------------------
# Amostra de usuários (se quiser usar todos, remova sample logic)
sample_size = 1000
unique_users = df_users["user_id"].drop_duplicates()
n_to_sample = min(sample_size, unique_users.nunique())
sample_users = unique_users.sample(n_to_sample, random_state=42).tolist()

entity_rows = []
for uid in sample_users:
    sub = df_users[df_users["user_id"] == uid]
    if sub.empty:
        continue
    label_ts = pd.to_datetime(sub[ts_col].max())  # último snapshot timestamp para aquele user
    entity_rows.append({"user_id": int(uid), "event_timestamp": label_ts})

entity_df = pd.DataFrame(entity_rows)
print("entity_df sample:", entity_df.head())

# ------------------ FUNÇÃO DE JANELAS (sem vazamento) ------------------
def compute_window_features_for_row(uid, label_ts, events_df, snapshot_df):
    """
    Retorna dict com features calculadas somente com eventos < label_ts.
    janelas: 7d, 30d, 60d. Se não houver events_df, usa snapshot_df como fallback.
    """
    res = {}
    windows = {"7d": 7, "30d": 30, "60d": 60}

    if events_df is not None:
        user_ev = events_df[events_df["user_id"] == uid].sort_values("event_ts")
        prev = user_ev[user_ev["event_ts"] < label_ts]

        if prev.shape[0] == 0:
            res["last_order_ts"] = pd.NaT
            res["days_since_last_order"] = np.nan
        else:
            last = prev["event_ts"].max()
            res["last_order_ts"] = last
            res["days_since_last_order"] = (label_ts - last).days

        for k, days in windows.items():
            start = label_ts - pd.Timedelta(days=days)
            window_df = prev[(prev["event_ts"] >= start) & (prev["event_ts"] < label_ts)]
            # assumimos 'is_purchase' se houver, senão contamos linhas
            if "is_purchase" in window_df.columns:
                res[f"num_orders_{k}"] = int(window_df["is_purchase"].sum())
            else:
                res[f"num_orders_{k}"] = int(window_df.shape[0])

            if "order_value" in window_df.columns and window_df.shape[0] > 0:
                res[f"avg_order_value_{k}"] = float(window_df["order_value"].mean())
            else:
                res[f"avg_order_value_{k}"] = np.nan

        # pct dias ativos últimos 60 dias
        last_60 = prev[(prev["event_ts"] >= label_ts - pd.Timedelta(days=60)) & (prev["event_ts"] < label_ts)]
        if last_60.shape[0] > 0:
            distinct_days = last_60["event_ts"].dt.floor("D").nunique()
            res["pct_active_60d"] = distinct_days / 60.0
        else:
            res["pct_active_60d"] = 0.0

    else:
        # fallback using snapshots in snapshot_df; less precise but works
        user_hist = snapshot_df[snapshot_df["user_id"] == uid].sort_values(ts_col)
        prev = user_hist[user_hist[ts_col] < label_ts]
        if prev.shape[0] == 0:
            res["last_order_ts"] = pd.NaT
            res["days_since_last_order"] = np.nan
            res["num_orders_7d"] = 0
            res["num_orders_30d"] = 0
            res["avg_order_value_7d"] = np.nan
            res["avg_order_value_30d"] = np.nan
            res["pct_active_60d"] = 0.0
        else:
            last = pd.to_datetime(prev[ts_col].max())
            res["last_order_ts"] = last
            res["days_since_last_order"] = (label_ts - last).days
            # use latest snapshot prior to label for approximate orders_30d / avg_value_30d
            last_snap = prev.loc[prev[ts_col].idxmax()]
            res["num_orders_30d"] = int(last_snap.get("orders_30d", 0))
            res["avg_order_value_30d"] = float(last_snap.get("avg_order_value", np.nan))
            # 7d approximations if available
            res["num_orders_7d"] = int(last_snap.get("orders_7d", 0)) if "orders_7d" in prev.columns else 0
            res["avg_order_value_7d"] = np.nan
            res["pct_active_60d"] = 0.0

    return res

# ------------------ Calcular derived features para cada linha de entity_df ------------------
derived_rows = []
for _, row in entity_df.iterrows():
    uid = int(row["user_id"])
    label_ts = pd.to_datetime(row["event_timestamp"])
    feats = compute_window_features_for_row(uid, label_ts, events if use_events else None, df_users)
    feats["user_id"] = uid
    feats["event_timestamp"] = label_ts
    derived_rows.append(feats)

derived_df = pd.DataFrame(derived_rows)
# garantir colunas esperadas
expected_cols = [
    "user_id", "event_timestamp", "days_since_last_order", "last_order_ts",
    "num_orders_7d", "num_orders_30d",
    "avg_order_value_7d", "avg_order_value_30d",
    "pct_active_60d"
]
for c in expected_cols:
    if c not in derived_df.columns:
        derived_df[c] = np.nan

print("Derived features sample:\n", derived_df.head())

# ------------------ Puxar features históricas do Feast e juntar ------------------
fs = FeatureStore(repo_path=str(feature_repo_path))
feature_list = [
    "user_stats:orders_30d",
    "user_stats:avg_order_value",
    "user_stats:last_order_ts",
]
fs_df = fs.get_historical_features(entity_df=entity_df, features=feature_list).to_df()
print("Feast returned columns:", fs_df.columns.tolist())
# normalizar nomes se vier com prefixo
rename_map = {}
if "user_stats:orders_30d" in fs_df.columns:
    rename_map["user_stats:orders_30d"] = "orders_30d"
if "user_stats:avg_order_value" in fs_df.columns:
    rename_map["user_stats:avg_order_value"] = "avg_order_value"
if "user_stats:last_order_ts" in fs_df.columns:
    rename_map["user_stats:last_order_ts"] = "last_order_ts_fs"
if rename_map:
    fs_df = fs_df.rename(columns=rename_map)

# merge: alinha por user_id + event_timestamp
full_df = pd.merge(fs_df, derived_df, how="left", on=["user_id", "event_timestamp"])
full_df = full_df.drop_duplicates(subset=["user_id", "event_timestamp"], keep="last")

# ------------------ LABEL (proxy) ------------------
# Default: churn proxy = sem pedidos nos últimos 30 dias (usar num_orders_30d quando possível)
if "label" not in full_df.columns:
    source_for_orders = None
    if "orders_30d" in full_df.columns:
        source_for_orders = "orders_30d"
    elif "num_orders_30d" in full_df.columns:
        source_for_orders = "num_orders_30d"

    if source_for_orders:
        full_df["label"] = (full_df[source_for_orders].fillna(0) == 0).astype(int)
    else:
        # fallback conservador: marca tudo como não-churn (0) para evitar vazamento
        full_df["label"] = 0
        print("Aviso: não foi possível inferir coluna de orders para gerar label. Labels preenchidos com 0 (ajuste a regra).")

# ------------------ LIMPEZA DE TIPOS E SALVAMENTO ------------------
# converter timestamps para timezone-aware e garantir tipos numéricos
if "event_timestamp" in full_df.columns:
    full_df["event_timestamp"] = pd.to_datetime(full_df["event_timestamp"], utc=True)
if "last_order_ts" in full_df.columns:
    full_df["last_order_ts"] = pd.to_datetime(full_df["last_order_ts"], utc=True)
if "last_order_ts_fs" in full_df.columns:
    full_df["last_order_ts_fs"] = pd.to_datetime(full_df["last_order_ts_fs"], utc=True)

# salvar parquet final
full_df.to_parquet(output_path, index=False)
print("Saved training dataset with derived features to", output_path)
print(full_df.head())
print("Columns saved:", full_df.columns.tolist())
