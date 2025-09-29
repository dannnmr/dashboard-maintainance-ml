from pathlib import Path
import pandas as pd

ESTADO_EMOJI = {"NORMAL": "✅", "ALERTA": "⚠️", "CRITICO": "🚨"}

def _try_import_deltalake():
    try:
        from deltalake import DeltaTable  # type: ignore
        return DeltaTable
    except Exception:
        return None

def cargar_datos_transformador_preprocesados(ruta_silver: Path, ruta_processed: Path) -> dict:
    """
    Lee Silver (Delta) y, si no, hace fallback a processed/*.parquet.
    Retorna dict con df_principal, df_anomalias (si existe o derivado), metadatos, info_variables.
    """
    print(" CARGA DE DATOS PREPROCESADOS DE TRANSFORMADORES (Delta First)")
    print("=" * 70)

    archivos_esperados = {
        "principal": "transformador_data.parquet",
        "anomalias": "transformador_data_anomalias.parquet",
        "metadatos": "transformador_data_metadatos_tecnicos.txt",
        "variables": "transformador_data_variables.csv",
    }

    # 1) Silver (Delta)
    df_principal = None
    if ruta_silver.exists():
        DeltaTable = _try_import_deltalake()
        if DeltaTable is not None:
            try:
                print(f"   Leyendo Silver (Delta): {ruta_silver}")
                dt = DeltaTable(str(ruta_silver))
                df_principal = dt.to_pandas()
            except Exception as e:
                print(f"  No se pudo leer Delta Silver: {e}")
        else:
            print("   Paquete 'deltalake' no disponible; saltando lectura Delta.")

    # 2) Fallback processed
    if df_principal is None:
        print("   Fallback a archivos en processed/")
        ruta_pq = ruta_processed / archivos_esperados["principal"]
        if not ruta_pq.exists():
            raise FileNotFoundError("No se encontró Silver Delta ni el parquet principal en processed/")
        df_principal = pd.read_parquet(ruta_pq)

    # 3) Índice temporal
    if "timestamp" in df_principal.columns:
        df_principal["timestamp"] = pd.to_datetime(df_principal["timestamp"], utc=True, errors="coerce")
        df_principal = df_principal.sort_values("timestamp").set_index("timestamp")
    elif not isinstance(df_principal.index, pd.DatetimeIndex):
        df_principal.index = pd.to_datetime(df_principal.index, utc=True, errors="coerce")
        df_principal = df_principal.sort_index()

    print(f"   * Dimensiones: {len(df_principal):,} × {df_principal.shape[1]}")
    if len(df_principal):
        print(f"   - Período: {df_principal.index.min()} → {df_principal.index.max()}")
        print(f"   - Duración: {df_principal.index.max() - df_principal.index.min()}")

    # 4) Anomalías (opcional parquet) o derivado
    df_anomalias = None
    ruta_anom = ruta_processed / archivos_esperados["anomalias"]
    if ruta_anom.exists():
        try:
            print("   Cargando dataset de anomalías (processed/parquet)...")
            df_anomalias = pd.read_parquet(ruta_anom)
            if "timestamp" in df_anomalias.columns:
                df_anomalias["timestamp"] = pd.to_datetime(df_anomalias["timestamp"], utc=True, errors="coerce")
                df_anomalias = df_anomalias.sort_values("timestamp").set_index("timestamp")
            elif not isinstance(df_anomalias.index, pd.DatetimeIndex):
                df_anomalias.index = pd.to_datetime(df_anomalias.index, utc=True, errors="coerce")
                df_anomalias = df_anomalias.sort_index()
        except Exception as e:
            print(f"   Error leyendo anomalías parquet: {e}")

    if df_anomalias is None and "estado_operacional" in df_principal.columns:
        print("   Derivando anomalías desde df_principal (ALERTA/CRITICO)...")
        df_anomalias = df_principal[df_principal["estado_operacional"].isin(["ALERTA", "CRITICO"])].copy()

    # 5) Metadatos (txt) e info variables (csv), opcionales
    metadatos = None
    ruta_meta = ruta_processed / archivos_esperados["metadatos"]
    if ruta_meta.exists():
        try:
            metadatos = ruta_meta.read_text(encoding="utf-8")
            print("\nMetadatos técnicos cargados")
        except Exception as e:
            print(f"   Error cargando metadatos: {e}")

    info_variables = None
    ruta_vars = ruta_processed / archivos_esperados["variables"]
    if ruta_vars.exists():
        try:
            info_variables = pd.read_csv(ruta_vars)
            print("\n🔧 Información de variables:")
            if "categoria_tecnica" in info_variables.columns:
                cats = info_variables.groupby("categoria_tecnica")["variable"].count()
                for categoria, qty in cats.items():
                    print(f"   • {categoria}: {qty} variables")
        except Exception as e:
            print(f"   Error cargando info de variables: {e}")

    # 6) Distribución de estados (si existe)
    if "estado_operacional" in df_principal.columns:
        dist = df_principal["estado_operacional"].value_counts(dropna=False)
        print("\n Distribución de estados operacionales:")
        for estado, cantidad in dist.items():
            total = len(df_principal) or 1
            porcentaje = cantidad * 100.0 / total
            emoji = ESTADO_EMOJI.get(str(estado), "📊")
            print(f"   {emoji} {estado}: {cantidad:,} ({porcentaje:.1f}%)")

    print("\n CARGA COMPLETADA — datos listos para Feature Engineering")
    return {
        "df_principal": df_principal,
        "df_anomalias": df_anomalias,
        "metadatos": metadatos,
        "info_variables": info_variables,
    }
