# Batch/online inference helper
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from paths import ARTIFACTS_DIR
from config import LOOKBACK
from ensemble import minmax_transform, smooth_alerts

def load_artifacts():
    medians = joblib.load(ARTIFACTS_DIR / "medians.pkl")
    ae = tf.keras.models.load_model(ARTIFACTS_DIR / "ae_lstm.keras")
    scaler_ae = joblib.load(ARTIFACTS_DIR / "scaler_ae.pkl")
    feature_cols = pd.read_csv(ARTIFACTS_DIR / "feature_columns.csv", header=None)[0].tolist()
    with open(ARTIFACTS_DIR / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return ae, scaler_ae, feature_cols, meta, medians

def make_sequence_from_window(df_window: pd.DataFrame, feature_cols: list[str]):
    X = df_window[feature_cols].astype(float)
    return X.values.astype(np.float32)  # shape (LOOKBACK, F)

def ae_score_from_window(ae, seq_batch: np.ndarray, ae_min: float, ae_max: float):
    # seq_batch shape: (1, LOOKBACK, F)
    rec = ae.predict(seq_batch, verbose=0)
    err = np.mean((seq_batch - rec)**2, axis=(1,2))  # shape (1,)
    score = minmax_transform(err, ae_min, ae_max)    # normalized [0,1]
    return float(score[0])

def infer_from_last_24h(df_last_24h: pd.DataFrame):
    ae, scaler_ae, feature_cols, meta, medians = load_artifacts()
    # scale with saved scaler
    X = df_last_24h[feature_cols].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(medians)  # << add this
    X_sc = scaler_ae.transform(X)
    seq = X_sc[-LOOKBACK:]  # ensure correct length
    seq = np.expand_dims(seq, 0)  # (1, L, F)
    score = ae_score_from_window(ae, seq, meta["ae_score_min"], meta["ae_score_max"])
    pred = int(score > meta["operate_thr"])
    return {"score": score, "pred": pred, "operate_thr": meta["operate_thr"]}
