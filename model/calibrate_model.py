#!/usr/bin/env python3
"""
model/calibrate_model.py

Script separado para calibrar probabilidades do modelo.
- Executa split temporal igual ao train script;
- Treina um LGBMClassifier (sklearn API) em uma sub-amostra do treino;
- Usa CalibratedClassifierCV(cv='prefit') com método automático:
    * 'isotonic' se >=200 positivos em calib set
    * caso contrário 'sigmoid'
- Avalia Brier + AUC antes/depois no conjunto de validação temporal;
- Salva calibrator e scaler em model/artifacts/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier

# Paths
repo_root = Path(__file__).resolve().parent.parent
data_path = repo_root / "data" / "training_dataset.parquet"
artifacts_dir = repo_root / "model" / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Outputs
calibrated_model_path = artifacts_dir / "churn_baseline_v2_calibrated.pkl"
scaler_path = artifacts_dir / "scaler_v2.pkl"
report_path = artifacts_dir / "calibration_report.json"

RANDOM_STATE = 42

def load_data():
    if not data_path.exists():
        raise SystemExit(f"Arquivo não encontrado: {data_path}")
    df = pd.read_parquet(data_path)
    if "event_timestamp" not in df.columns or "label" not in df.columns:
        raise SystemExit("O dataset precisa ter 'event_timestamp' e 'label'.")
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    return df

def temporal_split(df, val_quantile=0.8):
    cutoff = df["event_timestamp"].quantile(val_quantile)
    train_df = df[df["event_timestamp"] <= cutoff].copy()
    val_df = df[df["event_timestamp"] > cutoff].copy()
    return train_df, val_df, cutoff

def choose_features(df):
    # tenta carregar feature list salvo; fallback para heurística
    feat_file = artifacts_dir / "feature_list_v2.json"
    if feat_file.exists():
        try:
            features = json.load(open(feat_file))
            # garantia: filtrar colunas que existam no df
            features = [f for f in features if f in df.columns]
            if features:
                return features
        except Exception:
            pass
    # fallback heurística (as usadas no seu projeto)
    candidates = [
        "avg_order_value_7d","avg_order_value_30d","pct_active_60d",
        "orders_30d","avg_order_value","num_orders_60d"
    ]
    return [c for c in candidates if c in df.columns]

def train_and_calibrate(train_df, val_df, features):
    X_train_full = train_df[features].fillna(0).astype(float)
    y_train_full = train_df["label"].astype(int)
    X_val = val_df[features].fillna(0).astype(float)
    y_val = val_df["label"].astype(int)

    # split X_train_full into train_sub and calib (stratify when possible)
    strat = y_train_full if y_train_full.nunique() > 1 else None
    X_tr_sub, X_calib, y_tr_sub, y_calib = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=strat
    )

    # scaler: try to reuse existing scaler, if not, fit new one on X_tr_sub and save
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_tr_sub_scaled = scaler.transform(X_tr_sub)
        X_calib_scaled = scaler.transform(X_calib)
        X_val_scaled = scaler.transform(X_val)
    else:
        scaler = RobustScaler()
        X_tr_sub_scaled = scaler.fit_transform(X_tr_sub)
        X_calib_scaled = scaler.transform(X_calib)
        X_val_scaled = scaler.transform(X_val)
        joblib.dump(scaler, scaler_path)
        print("Saved scaler to", scaler_path)

    # train sklearn LGBMClassifier on sub-train
    n_estimators = 200
    clf = LGBMClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_tr_sub_scaled, y_tr_sub)

    # baseline probs (before calibration) on val
    probs_before = clf.predict_proba(X_val_scaled)[:,1]
    auc_before = roc_auc_score(y_val, probs_before) if y_val.nunique() > 1 else float("nan")
    brier_before = brier_score_loss(y_val, probs_before)

    # choose calibration method
    positives_calib = int(y_calib.sum())
    method = "isotonic" if positives_calib >= 200 else "sigmoid"
    print(f"Calibrating with method='{method}' (positives in calib={positives_calib})")

    calibrator = CalibratedClassifierCV(base_estimator=clf, method=method, cv="prefit")
    calibrator.fit(X_calib_scaled, y_calib)

    # evaluate calibrated
    probs_after = calibrator.predict_proba(X_val_scaled)[:,1]
    auc_after = roc_auc_score(y_val, probs_after) if y_val.nunique() > 1 else float("nan")
    brier_after = brier_score_loss(y_val, probs_after)

    # calibration curve (10 bins)
    prob_true, prob_pred = calibration_curve(y_val, probs_after, n_bins=10)

    # save calibrator
    joblib.dump(calibrator, calibrated_model_path)
    print("Saved calibrated model to", calibrated_model_path)

    report = {
        "method": method,
        "positives_in_calib": positives_calib,
        "auc_before": float(auc_before),
        "auc_after": float(auc_after),
        "brier_before": float(brier_before),
        "brier_after": float(brier_after),
        "calibration_curve": {
            "prob_true": [float(x) for x in prob_true.tolist()],
            "prob_pred": [float(x) for x in prob_pred.tolist()]
        },
        "features_used": features
    }
    json.dump(report, open(report_path, "w"), indent=2)
    print("Saved calibration report to", report_path)
    return report

def main():
    print("Loading data...")
    df = load_data()
    print("Temporal split (train/val)...")
    train_df, val_df, cutoff = temporal_split(df)
    print("Train rows:", len(train_df), "Val rows:", len(val_df), "cutoff:", cutoff)
    features = choose_features(df)
    if not features:
        raise SystemExit("Nenhuma feature detectada para calibrar. Ajuste feature_list_v2.json ou candidates.")
    print("Using features:", features)

    report = train_and_calibrate(train_df, val_df, features)
    print("Calibration finished. Summary:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
