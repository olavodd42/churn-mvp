# scripts/ts_cv.py
import pandas as pd, numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib
import lightgbm as lgb

df = pd.read_parquet("data/training_dataset.parquet")
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
df = df.sort_values("event_timestamp")

features = ["avg_order_value_7d","avg_order_value_30d","pct_active_60d","orders_30d","avg_order_value"]  # adapte
X = df[features].fillna(0).astype(float)
y = df["label"].astype(int)

tss = TimeSeriesSplit(n_splits=5)
aucs=[]
for train_idx, val_idx in tss.split(X):
    Xtr, Xv = X.iloc[train_idx], X.iloc[val_idx]
    ytr, yv = y.iloc[train_idx], y.iloc[val_idx]
    dtr = lgb.Dataset(Xtr, label=ytr)
    dv = lgb.Dataset(Xv, label=yv, reference=dtr)
    bst = lgb.train({ "objective":"binary","metric":"auc","verbosity":-1, "seed":42}, dtr, num_boost_round=200)
    aucs.append(roc_auc_score(yv, bst.predict(Xv)))
print("TS CV AUCs:", aucs, "mean:", np.mean(aucs))
