# app/model_loader.py
# Loads artifacts (scalers, models) and provides a unified interface.

import os
import json
import pandas as pd
from typing import Dict, List, Any

# ---- Correct imports for unpickling ----
import joblib  # <-- use joblib.load for files saved with joblib.dump
from tensorflow.keras.models import load_model  # <-- to load ae_lstm.keras

# scikit-learn imports (for compatibility when unpickling)
try:
    import sklearn
    import sklearn.ensemble
    import sklearn.preprocessing
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler, LabelEncoder
except Exception as e:
    print(f"Warning: sklearn import failed: {e}")
    sklearn = None


class ModelBundle:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.feature_columns = self._load_feature_columns()
        self.meta = self._try_load_json("meta.json")

        # All these were saved with joblib.dump(...)
        self.iforest   = self._load_joblib("iforest.pkl")
        self.scaler_if = self._load_joblib("scaler_if.pkl")

        # AE model and scaler
        self.ae_model  = self._load_ae("ae_lstm.keras")
        self.scaler_ae = self._load_joblib("scaler_ae.pkl")

        # Optional label encoder
        self.label_encoder = self._try_load_joblib("label_encoder.pkl")

    def _p(self, fname: str) -> str:
        return os.path.join(self.model_dir, fname)

    def _load_feature_columns(self) -> List[str]:
        path = self._p("feature_columns.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"feature_columns.csv not found at {path}")
        s = pd.read_csv(path, header=None).iloc[:, 0].tolist()
        return [str(x) for x in s]

    # ---- Use joblib.load (not pickle.load) ----
    def _load_joblib(self, fname: str):
        path = self._p(fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{fname} not found at {path}")
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Error loading {fname} with joblib: {e}")
            raise

    def _try_load_joblib(self, fname: str):
        try:
            return self._load_joblib(fname)
        except Exception:
            return None

    def _try_load_json(self, fname: str) -> Dict[str, Any]:
        path = self._p(fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_ae(self, fname: str):
        path = self._p(fname)
        if os.path.exists(path):
            try:
                return load_model(path)
            except Exception as e:
                print(f"Error loading AE model {fname}: {e}")
                return None
        return None
