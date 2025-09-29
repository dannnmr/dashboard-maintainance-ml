# app/model_loader.py
# Loads artifacts (scalers, models) and provides a unified interface.
# Based on your ZIP: ae_lstm.keras, iforest.pkl, scaler_if.pkl, scaler_ae.pkl, label_encoder.pkl, feature_columns.csv

import os
import json
import pickle
import pandas as pd
from typing import Dict, List, Any

# Import sklearn modules that might be needed for unpickling
try:
    import sklearn
    import sklearn.ensemble
    import sklearn.preprocessing
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler, LabelEncoder
except ImportError as e:
    print(f"Warning: sklearn import failed: {e}")
    sklearn = None


    load_model = None

class ModelBundle:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.feature_columns = self._load_feature_columns()
        self.meta = self._try_load_json("meta.json")

        self.iforest = self._load_pickle("iforest.pkl")
        self.scaler_if = self._load_pickle("scaler_if.pkl")

        self.ae_model = self._load_ae("ae_lstm.keras")
        self.scaler_ae = self._load_pickle("scaler_ae.pkl")

        # Optional label encoder (e.g., for status labels)
        self.label_encoder = self._try_load_pickle("label_encoder.pkl")

    def _p(self, fname: str) -> str:
        return os.path.join(self.model_dir, fname)

    def _load_feature_columns(self) -> List[str]:
        path = self._p("feature_columns.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"feature_columns.csv not found at {path}")
        s = pd.read_csv(path, header=None).iloc[:, 0].tolist()
        return [str(x) for x in s]

    def _load_pickle(self, fname: str):
        path = self._p(fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{fname} not found at {path}")
        
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except ModuleNotFoundError as e:
            print(f"ModuleNotFoundError loading {fname}: {e}")
            print("Available sklearn modules:", dir(sklearn) if sklearn else "sklearn not available")
            raise
        except Exception as e:
            print(f"Error loading pickle file {fname}: {e}")
            raise

    def _try_load_pickle(self, fname: str):
        try:
            return self._load_pickle(fname)
        except Exception:
            return None

    def _try_load_json(self, fname: str) -> Dict[str, Any]:
        path = self._p(fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_ae(self, fname: str):
        if load_model is None:
            return None
        path = self._p(fname)
        if os.path.exists(path):
            return load_model(path)
        return None
