# etl/bronze/storage.py
from logger_bronze import logger
import pandas as pd
import polars as pl
import numpy as np
from typing import Optional
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from config_bronze import BRONZE_TABLE  # Path to the Delta table directory

def leer_ultimo_timestamp(tag_alias: str) -> Optional[pd.Timestamp]:
    """
    Read MAX(timestamp) for a given tag directly from Delta Lake.
    Returns None if table does not exist or tag has no rows with valid timestamp.
    """
    try:
        dt = DeltaTable(str(BRONZE_TABLE))
    except Exception:
        return None

    try:
        pa_tbl = dt.to_pyarrow_table(filters=[("tag", "=", tag_alias)], columns=["timestamp", "tag"])
    except TypeError:
        pa_tbl = dt.to_pyarrow_table(columns=["timestamp", "tag"])

    if pa_tbl.num_rows == 0:
        return None

    pdf = pa_tbl.to_pandas()
    if "tag" in pdf.columns:
        pdf = pdf[pdf["tag"] == tag_alias]

    if pdf.empty:
        return None

    ts = pd.to_datetime(pdf["timestamp"], utc=True, errors="coerce")
    if ts.notna().any():
        return ts.max().floor("s")
    return None


def _prepare_partitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare 'date' partition safely:
    - If timestamp is valid -> yyyy-mm-dd
    - If timestamp is NaT -> '__missing__'
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    date_str = np.where(
        ts.notna(),
        ts.dt.strftime("%Y-%m-%d"),
        "__missing__"
    )
    df["date"] = date_str.astype(str)
    return df


def guardar_bronze_delta(df_nuevo: pd.DataFrame, tag_alias: str) -> None:
    """
    Append rows to Bronze Delta without dropping anything.
    Expected columns: ['timestamp','value','value_text','value_bool','tag'] (+ derived 'date')
    - 'value'      : float (NaN allowed)
    - 'value_text' : string as-received (None allowed; dict/error -> None)
    - 'value_bool' : boolean state (None allowed)
    - 'timestamp'  : tz-aware datetime or NaT
    - partition_by : ['tag','date'] (date='__missing__' for NaT)
    """
    if df_nuevo is None or df_nuevo.empty:
        logger.info(f"Sin datos para guardar en {tag_alias}")
        return

    df = df_nuevo.copy()

    # Enforce dtypes (without dropping)
    df["timestamp"]  = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["value"]      = pd.to_numeric(df["value"], errors="coerce")           # float (NaN ok)
    # value_text: keep as object/string, but cast to string where not-null
    df["value_text"] = df["value_text"].where(df["value_text"].isna(), df["value_text"].astype(str))
    # value_bool: keep None/True/False; pandas may store as object; polars will cast to Boolean
    # tag always string
    df["tag"]        = df["tag"].astype(str)

    # Safe partitions
    df = _prepare_partitions(df)  # adds 'date'

    # Convert to Polars for delta-rs writer with proper types
    pl_df = pl.from_pandas(df[["timestamp", "value", "value_text", "value_bool", "tag", "date"]])

    # Ensure expected dtypes
    # timestamp
    if not isinstance(pl_df.schema.get("timestamp"), pl.Datetime):
        pl_df = pl_df.with_columns(
            pl.col("timestamp").cast(pl.Datetime(time_unit="us", time_zone="UTC"))
        )
    # value (float)
    if pl_df.schema.get("value") != pl.Float64:
        pl_df = pl_df.with_columns(pl.col("value").cast(pl.Float64))
    # value_text (Utf8)
    if pl_df.schema.get("value_text") != pl.Utf8:
        pl_df = pl_df.with_columns(pl.col("value_text").cast(pl.Utf8))
    # value_bool (Boolean)
    if pl_df.schema.get("value_bool") != pl.Boolean:
        pl_df = pl_df.with_columns(pl.col("value_bool").cast(pl.Boolean))
    # tag/date (Utf8)
    if pl_df.schema.get("tag") != pl.Utf8:
        pl_df = pl_df.with_columns(pl.col("tag").cast(pl.Utf8))
    if pl_df.schema.get("date") != pl.Utf8:
        pl_df = pl_df.with_columns(pl.col("date").cast(pl.Utf8))

    try:
        write_deltalake(
            str(BRONZE_TABLE),
            pl_df,
            mode="append",
            partition_by=["tag", "date"],
            schema_mode="merge",  # allow adding value_text/value_bool safely
        )
        logger.info(f"Datos guardados en Delta Lake (tag={tag_alias}).")
    except Exception as e:
        logger.error(f"Error escribiendo Delta Lake para {tag_alias}: {e}")
        raise
