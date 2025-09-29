# app/utils.py
# Utilities to ensure feature ordering and scaling.

import numpy as np
import pandas as pd
from typing import List, Dict

def ensure_dataframe(records: list[dict], feature_order: List[str]) -> pd.DataFrame:
    # Normalize keys to string and align columns
    df = pd.DataFrame(records)
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        # create missing with NaN
        for m in missing:
            df[m] = np.nan
    # reorder
    df = df[feature_order]
    return df
