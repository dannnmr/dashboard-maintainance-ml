import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def fit_iforest(X_train_normal: pd.DataFrame, random_state=42):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_normal)
    model = IsolationForest(
        n_estimators=400, contamination='auto',
        random_state=random_state, n_jobs=-1
    ).fit(X_tr)
    return model, scaler

def iforest_scores(model: IsolationForest, scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    X_sc = scaler.transform(X)
    return -model.score_samples(X_sc)  # higher = more anomalous
