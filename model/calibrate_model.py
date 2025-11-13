#!/usr/bin/env python3
"""
model/calibrate_model.py — versão robusta

Melhorias:
- remove features constantes / quase-constantes antes do treino
- checa se sub-train / calib têm pelo menos 2 classes; adapta estratégia
- usa LGBMClassifier com parâmetros permissivos quando necessário
- alterna entre cv='prefit' (quando calib tem >50 positivos) ou cv=5 (quando calib pequeno)
- salva scaler + calibrator + relatório JSON
- imprime diagnósticos úteis
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier

# Paths
repo_root = Path(__file__).resolve().parent.parent
data_path = repo_root / "data" / "training_dataset.parquet"
artifacts_dir = repo_root / "model" / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)

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
    feat_file = artifacts_dir / "feature_list_v2.json"
    if feat_file.exists():
        try:
            features = json.load(open(feat_file))
            features = [f for f in features if f in df.columns]
            if features:
                return features
        except Exception:
            pass
    candidates = [
        "avg_order_value_7d","avg_order_value_30d","avg_order_value_60d",
        "pct_active_60d","orders_30d","avg_order_value","num_orders_60d"
    ]
    return [c for c in candidates if c in df.columns]

def drop_low_variance(X, tol_zero_frac=0.99, tol_std=1e-6):
    """Remove colunas quase-constantes."""
    to_drop = []
    for c in X.columns:
        col = X[c]
        if col.isna().all():
            to_drop.append(c)
            continue
        if (col == col.iloc[0]).all():
            to_drop.append(c)
            continue
        zero_frac = (col == 0).mean()
        if zero_frac >= tol_zero_frac:
            to_drop.append(c)
            continue
        if pd.api.types.is_numeric_dtype(col):
            if col.std() < tol_std:
                to_drop.append(c)
    return [c for c in X.columns if c not in to_drop], to_drop

def train_and_calibrate(train_df, val_df, features):
    # prepare matrices
    X_train_full = train_df[features].copy()
    y_train_full = train_df["label"].astype(int).copy()
    X_val = val_df[features].copy()
    y_val = val_df["label"].astype(int).copy()

    # basic diagnostics
    print("=== Diagnostics ===")
    print("Train shape:", X_train_full.shape, "Positives:", int(y_train_full.sum()), "Negatives:", int((~y_train_full.astype(bool)).sum()))
    print("Val shape:", X_val.shape, "Positives:", int(y_val.sum()), "Negatives:", int((~y_val.astype(bool)).sum()))
    print("Feature uniques / na counts:")
    for c in features:
        print(f" {c:25} | nunique={X_train_full[c].nunique():4}  null_frac={(X_train_full[c].isna().mean()):.3f}")

    # impute simple
    X_train_full = X_train_full.fillna(0).astype(float)
    X_val = X_val.fillna(0).astype(float)

    # drop low-variance cols
    kept, dropped = drop_low_variance(X_train_full, tol_zero_frac=0.995, tol_std=1e-8)
    if dropped:
        print("Dropped low-variance features:", dropped)
        X_train_full = X_train_full[kept]
        X_val = X_val[kept]
        features = kept

    if X_train_full.shape[1] == 0:
        raise SystemExit("Nenhuma feature restante após remoção de baixa variância. Ajuste features.")

    # split train -> subtrain + calib
    strat = y_train_full if y_train_full.nunique() > 1 else None
    X_tr_sub, X_calib, y_tr_sub, y_calib = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=strat
    )

    # verify classes in subtrain/calib
    print("Sub-train pos/neg:", int(y_tr_sub.sum()), (len(y_tr_sub) - int(y_tr_sub.sum())))
    print("Calib   pos/neg:", int(y_calib.sum()), (len(y_calib) - int(y_calib.sum())))
    if y_tr_sub.nunique() < 2:
        raise SystemExit("Sub-train tem apenas uma classe — ajuste split/labels.")
    if y_calib.nunique() < 2:
        print("Aviso: calib tem apenas uma classe; método 'prefit' NÃO será usado — faremos CV no treino completo.")

    # scaler
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_tr_sub_s = scaler.transform(X_tr_sub)
        X_calib_s = scaler.transform(X_calib)
        X_val_s = scaler.transform(X_val)
    else:
        scaler = RobustScaler()
        X_tr_sub_s = scaler.fit_transform(X_tr_sub)
        X_calib_s = scaler.transform(X_calib)
        X_val_s = scaler.transform(X_val)
        joblib.dump(scaler, scaler_path)
        print("Saved scaler to", scaler_path)

    # dataframe wrappers (keep column names to avoid "invalid feature names" warnings)
    X_tr_sub_s_df = pd.DataFrame(X_tr_sub_s, columns=features, index=X_tr_sub.index)
    X_calib_s_df = pd.DataFrame(X_calib_s, columns=features, index=X_calib.index)
    X_val_s_df = pd.DataFrame(X_val_s, columns=features, index=X_val.index)
    X_train_full_s_df = pd.DataFrame(scaler.transform(X_train_full), columns=features, index=X_train_full.index)

    # prepare LGBMClassifier with permissive fallback params
    base_params = dict(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        num_leaves=31,          # moderado; maior -> potencial overfit
        max_depth=6,            # evita árvores muito profundas
        min_child_samples=5,    # evita folhas com poucos exemplos; reduzir para 1 se dataset pequeno
        min_split_gain=0.0,     # permite splits com ganho pequeno (evita -inf)
        min_child_weight=1e-3,  # protege contra divisões numéricas
        reg_alpha=0.0,
        reg_lambda=0.0,
        max_bin=255,            # controle do histograma; às vezes ajuda
        force_row_wise=True,    # evita overhead col-wise e pode reduzir warnings em Windows
        verbosity=-1,
    )

    # if very small calib or features discrete -> relax further
    if int(y_tr_sub.sum()) < 10 or X_tr_sub_s_df.shape[1] <= 3:
        base_params.update({"min_child_samples": 1, "num_leaves": 63, "min_split_gain": 0.0})

    clf = LGBMClassifier(**base_params)
    # fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr_sub_s_df, y_tr_sub)

    # baseline probs
    probs_before = clf.predict_proba(X_val_s_df)[:, 1]
    auc_before = roc_auc_score(y_val, probs_before) if y_val.nunique() > 1 else float("nan")
    brier_before = brier_score_loss(y_val, probs_before)

    # choose calibration strategy
    positives_calib = int(y_calib.sum())
    method = "isotonic" if positives_calib >= 200 else "sigmoid"
    use_prefit = (positives_calib >= 50 and y_calib.nunique() > 1)
    print(f"Calib positives: {positives_calib}, method={method}, use_prefit={use_prefit}")

    if use_prefit:
        # prefit: fit calibrator on held-out calib
        calibrator = CalibratedClassifierCV(estimator=clf, method=method, cv="prefit")
        calibrator.fit(X_calib_s_df, y_calib)
    else:
        # fallback: cross-validated calibration on full train (more robust when calib small)
        print("Using CalibratedClassifierCV(cv=5) on full train (mais robusto com calib pequeno).")
        base2 = LGBMClassifier(**base_params)
        calibrator = CalibratedClassifierCV(estimator=base2, method="sigmoid", cv=5)
        calibrator.fit(X_train_full_s_df, y_train_full)

    # evaluate after calibration
    probs_after = calibrator.predict_proba(X_val_s_df)[:, 1]
    auc_after = roc_auc_score(y_val, probs_after) if y_val.nunique() > 1 else float("nan")
    brier_after = brier_score_loss(y_val, probs_after)

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
        "dropped_features": dropped,
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
