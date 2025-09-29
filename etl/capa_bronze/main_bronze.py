# etl/bronze/main.py
from logger_bronze import logger
from config_bronze import TAGS, START_TIME, END_TIME
from extract_bronze import (
    obtener_webid,
    generar_rangos_fechas,
    obtener_datos_hist_pag,
)
from storage_bronze import leer_ultimo_timestamp, guardar_bronze_delta
import pandas as pd
import numpy as np

"""
Bronze collector (RAW, non-destructive):
- Do NOT drop rows.
- If PI returns a dict (error/no data), store as NULL (no JSON serialization).
- Keep three convenience columns:
    * value       : numeric (float), NaN if non-numeric or dict
    * value_text  : textual representation exactly as received (None if dict/error)
    * value_bool  : boolean if the original value is a bool, else None
- Keep invalid timestamps (as NaT). They will be partitioned under date="__missing__".
"""

def split_pi_value(x):
    """
    Split incoming PI value into numeric/text/bool channels without dropping rows.
    Rules:
      - dict (PI error/no data): value=None, value_text=None, value_bool=None
      - bool: value=NaN, value_text="True"/"False", value_bool=True/False
      - int/float: value=float(x), value_text=str(x), value_bool=None
      - str: value=float(x) if convertible (handles dot/comma), value_text=str(x), value_bool=None
      - None: all None/NaN
    """
    # dict => treat as NULL in all channels (your request)
    if isinstance(x, dict):
        return np.nan, None, None

    # booleans as state
    if isinstance(x, bool):
        return np.nan, str(x), bool(x)

    # numeric types
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            return float(x), str(x), None
        except Exception:
            return np.nan, str(x), None

    # strings and other scalar-like values
    if isinstance(x, str):
        # try to be friendly with comma decimal
        x_strip = x.strip()
        x_num = np.nan
        try:
            # First attempt raw float
            x_num = float(x_strip)
        except Exception:
            # Second attempt replacing comma by dot
            try:
                x_num = float(x_strip.replace(",", "."))
            except Exception:
                x_num = np.nan
        return x_num, x, None

    # None or unknown types
    if x is None:
        return np.nan, None, None

    # Fallback: record textual repr but keep numeric NaN and bool None
    try:
        return np.nan, str(x), None
    except Exception:
        return np.nan, None, None


def extraer_datos_actualizados(tag_alias: str, tag_path: str) -> None:
    try:
        webid = obtener_webid(tag_path)

        # Resume from last stored timestamp (inclusive), move +1s to avoid duplicates
        fecha_inicio = leer_ultimo_timestamp(tag_alias)
        if fecha_inicio is not None:
            fecha_inicio = fecha_inicio + pd.Timedelta(seconds=1)
            logger.info(f"Reanudando desde {fecha_inicio} para {tag_alias}")
        else:
            fecha_inicio = pd.to_datetime(START_TIME, utc=True)
            logger.info(f"Extrayendo desde cero para {tag_alias}")
        

        rangos = generar_rangos_fechas(str(fecha_inicio), END_TIME, delta_dias=15)
        df_total = pd.DataFrame()

        for inicio, fin in rangos:
            logger.info(f"[{tag_alias}] Rango: {inicio} -> {fin}")
            df = obtener_datos_hist_pag(webid, inicio, fin)  # columns: ['timestamp', 'value']
            if df.empty:
                continue

            # Normalize timestamp to tz-aware. DO NOT drop NaT.
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

            # Split channels
            # returns columns: value (float), value_text (str/None), value_bool (bool/None)
            out = df["value"].apply(split_pi_value).apply(pd.Series)
            out.columns = ["value", "value_text", "value_bool"]

            # Attach tag and pack
            df_pack = pd.concat([df[["timestamp"]], out], axis=1)
            df_pack["tag"] = tag_alias

            df_total = pd.concat([df_total, df_pack], ignore_index=True)

        if not df_total.empty:
            guardar_bronze_delta(df_total, tag_alias)
            logger.info(f"{tag_alias}: {len(df_total)} filas nuevas escritas en Bronze Delta.")
        else:
            logger.info(f"No se encontraron datos nuevos para {tag_alias}.")

    except Exception as e:
        logger.error(f"[ERROR] {tag_alias}: {e}")


if __name__ == "__main__":
    for tag_alias, tag_path in TAGS.items():
        logger.info(f"\nProcesando tag: {tag_alias}")
        extraer_datos_actualizados(tag_alias, tag_path)
