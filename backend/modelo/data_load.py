import pandas as pd
from deltalake import DeltaTable
from paths import RUTA_GOLD_COMPLETE

def load_gold_complete() -> pd.DataFrame:
    dt = DeltaTable(str(RUTA_GOLD_COMPLETE))
    df = dt.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").set_index("timestamp")
    return df
