from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import LEAK_OR_TEXT_COLS, TARGET
from paths import PLOTS_DIR

def select_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in numeric_cols if c not in LEAK_OR_TEXT_COLS]
    # drop constants and >30% NaN
    const = [c for c in cols if df[c].nunique() <= 1]
    cols = [c for c in cols if c not in const]
    nan_ratio = df[cols].isna().mean()
    high_nan = nan_ratio[nan_ratio > 0.30].index.tolist()
    cols = [c for c in cols if c not in high_nan]
    return cols

def temporal_split(X: pd.DataFrame, y: pd.Series, split_ratio=0.8):
    n = len(X); cut = int(n * split_ratio)
    return (X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:])

def fit_impute_train_medians(X_train: pd.DataFrame) -> pd.Series:
    return X_train.median(numeric_only=True)

def apply_impute(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(medians)

def build_label_encoder(y: pd.Series) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(y.astype(str))
    return le

def scale_fit_transform_normal(X_train: pd.DataFrame, y_train_enc, normal_id: int):
    mask_normal = (y_train_enc == normal_id)
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train[mask_normal]),
        index=X_train.index[mask_normal], columns=X_train.columns
    )
    return scaler, X_train_sc, mask_normal

def scale_transform(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

def make_sequences(X: pd.DataFrame, lookback: int, horizon_shift: int):
    Xv = X.values; N, F = Xv.shape
    seqs, idx = [], []
    for t in range(lookback-1, N-horizon_shift):
        seqs.append(Xv[t-(lookback-1):t+1])
        idx.append(X.index[t])  # end timestamp
    return np.array(seqs, dtype=np.float32), pd.Index(idx, name="timestamp")
