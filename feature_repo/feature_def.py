# feature_repo/feature_def.py
from datetime import timedelta
import sys
from pathlib import Path

try:
    import feast
    from feast import Entity, Field, FeatureView, FileSource, FeatureStore
except Exception as e:
    print("Erro ao importar feast:", e)
    sys.exit(1)

# detecta módulos/objetos disponíveis
HAS_FEAST_TYPES = hasattr(feast, "types")
HAS_VALUE_TYPE = False
try:
    from feast.value_type import ValueType as FeastValueType
    HAS_VALUE_TYPE = True
except Exception:
    FeastValueType = None

# funções separadas: uma para Entity.value_type (ValueType enum),
# outra para Field.dtype (feast.types.*)
def entity_value_type(kind: str):
    k = kind.lower()
    if HAS_VALUE_TYPE and FeastValueType is not None:
        if k == "int":
            return FeastValueType.INT64 if hasattr(FeastValueType, "INT64") else FeastValueType.INT32
        if k == "float":
            # prefer DOUBLE or FLOAT
            for c in ("DOUBLE", "FLOAT"):
                if hasattr(FeastValueType, c):
                    return getattr(FeastValueType, c)
            return FeastValueType.DOUBLE if hasattr(FeastValueType, "DOUBLE") else FeastValueType.FLOAT
        if k == "ts":
            return FeastValueType.UNIX_TIMESTAMP if hasattr(FeastValueType, "UNIX_TIMESTAMP") else FeastValueType.INT64
    # fallback python types
    if k == "int":
        return int
    if k == "float":
        return float
    if k == "ts":
        return int
    raise RuntimeError("Não foi possível mapear entity_value_type para " + kind)


def field_dtype(kind: str):
    k = kind.lower()
    if HAS_FEAST_TYPES:
        import feast.types as ft
        # floats
        if k == "float":
            for candidate in ("Float64", "Float32", "Float"):
                if hasattr(ft, candidate):
                    return getattr(ft, candidate)
        # ints
        if k == "int":
            for candidate in ("Int64", "Int32", "Int"):
                if hasattr(ft, candidate):
                    return getattr(ft, candidate)
        # timestamps
        if k == "ts":
            # Feast defines UnixTimestamp in feast.types
            if hasattr(ft, "UnixTimestamp"):
                return getattr(ft, "UnixTimestamp")
            for candidate in ("Int64", "Int32", "Int"):
                if hasattr(ft, candidate):
                    return getattr(ft, candidate)
    # fallbacks
    if k == "float":
        return float
    if k in ("int", "ts"):
        return int
    raise RuntimeError("Não foi possível mapear field_dtype para " + kind)


# paths
repo_root = Path(__file__).resolve().parent.parent
data_path = repo_root / "data" / "user_features.parquet"

if not data_path.exists():
    print("Aviso: arquivo esperado não encontrado:", data_path)
    print("Gere 'data/user_features.parquet' com scripts/prepare_user_features.py antes de materializar.")
    # não aborta; permite aplicar registry mesmo sem arquivo

# FileSource (offline store)
events = FileSource(
    path=str(data_path),
    # declarar ambos para compatibilidade com versões diferentes do Feast:
    event_timestamp_column="event_ts",   # usado por várias versões/guia do Feast
    timestamp_field="event_ts",          # campo explícito que remove ambiguidade
    created_timestamp_column=None,
)

# Entity usando ValueType enum (ou python fallback)
user = Entity(
    name="user_id",
    value_type=entity_value_type("int"),
    description="User id",
)

# FeatureView com Field(dtype=feast.types.*)
user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user],              # <--- CORREÇÃO: passar a variável Entity
    ttl=timedelta(days=90),
    schema=[
        Field(name="orders_30d", dtype=field_dtype("float")),
        Field(name="avg_order_value", dtype=field_dtype("float")),
        Field(name="last_order_ts", dtype=field_dtype("ts")),
    ],
    source=events,
    tags={"team": "churn"},
)

if __name__ == "__main__":
    fs = FeatureStore(repo_path=".")
    fs.apply([user, user_stats_fv])
    print("Applied entity & featureview to the registry.")
    print("Feast version detected:", feast.__version__)
    print("HAS_FEAST_TYPES:", HAS_FEAST_TYPES, "HAS_VALUE_TYPE:", HAS_VALUE_TYPE)
