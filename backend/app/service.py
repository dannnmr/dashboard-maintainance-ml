# app/service.py
# Wraps the inference logic to keep main.py thin.
# Combines AE reconstruction error + IsolationForest score into a final score/label.

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from app.model_loader import ModelBundle
from app.utils import ensure_dataframe

class AnomalyService:
    def __init__(self, model_dir: str):
        self.bundle = ModelBundle(model_dir)
        self.feature_columns = self.bundle.feature_columns
        self.meta = self.bundle.meta

    # ---- Public API ----
    def healthcheck(self) -> tuple[bool, Dict[str, Any]]:
        ok = True
        details = {
            "feature_columns": len(self.feature_columns),
            "iforest_loaded": self.bundle.iforest is not None,
            "scaler_if_loaded": self.bundle.scaler_if is not None,
            "ae_loaded": self.bundle.ae_model is not None,
            "scaler_ae_loaded": self.bundle.scaler_ae is not None,
        }
        ok = ok and details["iforest_loaded"] and details["scaler_if_loaded"]
        # AE is optional – if not available, we can serve IForest-only
        return ok, details

    def predict_from_records(self, records: List[Dict[str, float]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        df = ensure_dataframe(records, self.feature_columns)
        return self._predict_df(df)

    def predict_from_parquet(self, parquet_path: str, limit_rows: int = 200) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        df = pd.read_parquet(parquet_path)
        if limit_rows > 0:
            df = df.tail(limit_rows)
        # Keep only model features if parquet includes extra columns
        cols = [c for c in self.feature_columns if c in df.columns]
        df = df[cols]
        return self._predict_df(df)

    # ---- Internal ----
    def _predict_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        results: List[Dict[str, Any]] = []

        # Isolation Forest
        X_if = self.bundle.scaler_if.transform(df.values) if self.bundle.scaler_if else df.values
        if_scores = None
        if self.bundle.iforest:
            # decision_function: higher is less anomalous; we invert to get "anomaly score"
            if_scores = -self.bundle.iforest.decision_function(X_if)
        else:
            if_scores = np.zeros(len(df))

        # Autoencoder (optional)
        ae_scores = None
        if self.bundle.ae_model and self.bundle.scaler_ae:
            X_ae = self.bundle.scaler_ae.transform(df.values)
            X_hat = self.bundle.ae_model.predict(X_ae, verbose=0)
            # Reconstruction error as anomaly proxy
            ae_scores = np.mean(np.square(X_ae - X_hat), axis=1)
        else:
            ae_scores = np.zeros(len(df))

        # Simple ensemble: normalized average (you can change weights)
        s_if = (if_scores - if_scores.min()) / (if_scores.ptp() + 1e-8)
        s_ae = (ae_scores - ae_scores.min()) / (ae_scores.ptp() + 1e-8) if len(df) > 1 else ae_scores
        final_score = 0.5 * s_if + 0.5 * s_ae

        # Label by threshold (adjust or make it part of meta.json)
        thr = float(self.meta.get("threshold", 0.6))
        labels = np.where(final_score >= thr, "ANOMALY", "NORMAL")

        for i, (sc, lb) in enumerate(zip(final_score.tolist(), labels.tolist())):
            results.append({"index": int(df.index[i]) if hasattr(df.index, "__iter__") else i, "score": float(sc), "label": lb})

        return df, results
