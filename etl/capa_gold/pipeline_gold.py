# path: gold/pipeline_gold.py
from __future__ import annotations
from pathlib import Path
from paths_gold import RUTA_SILVER, RUTA_PROCESSED, RUTA_GOLD_BASE
from config_gold_v2 import PARAMETROS_TRANSFORMADOR
from lectures import cargar_datos_transformador_preprocesados
from validate_gold import validar_coherencia_tecnica_transformador
from features_thermal import crear_features_termicos_avanzados
from features_electrical import crear_features_electricos_avanzados
from labels_gold import crear_etiquetas_prediccion_transformador
from finalize_gold import finalizar_dataset_transformador

def run_gold_pipeline(
    ruta_silver: Path = RUTA_SILVER,
    ruta_processed: Path = RUTA_PROCESSED,
    ruta_features: Path = RUTA_GOLD_BASE,
    parametros: dict = PARAMETROS_TRANSFORMADOR,
    save_parquet_csv: bool = True,
    save_delta: bool = True,
) -> dict:
    datos = cargar_datos_transformador_preprocesados(ruta_silver, ruta_processed)
    df = datos["df_principal"]
    df_anom = datos["df_anomalias"]

    # Validación técnica (opcional, pero útil)
    _ = validar_coherencia_tecnica_transformador(df, parametros)

    # Features
    df_t, feats_t = crear_features_termicos_avanzados(df, parametros)
    df_e, feats_e = crear_features_electricos_avanzados(df_t, parametros)

    # Etiquetas
    df_l, etiquetas = crear_etiquetas_prediccion_transformador(df_e, df_anom, parametros)

    # Finalizar y guardar
    resultado = finalizar_dataset_transformador(
        df_l, feats_t, feats_e, etiquetas, ruta_features, parametros,
        save_parquet_csv=save_parquet_csv, save_delta=save_delta
    )
    return resultado
