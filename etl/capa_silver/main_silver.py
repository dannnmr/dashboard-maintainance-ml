import pandas as pd
from config_silver_v2 import RUTA_SILVER, RUTA_PROCESSED, RUTA_BRONZE, RANGOS_NORMALES
from logger_silver import get_logger
from bronze_silver import get_available_tags, cargar_datos_transformador_from_bronze
from transform_silver import (
consolidar_datos_transformador, limpiar_nombres_columnas_transformador,
convertir_tipos_transformador, analizar_valores_faltantes_transformador, tratar_valores_faltantes_transformador
)
from rules_silver import definir_criterios_transformador
from classify_silver import clasificar_estados_operacionales
from report_silver import generar_reporte_calidad_transformador
from save_silver import guardar_dataset_transformador

log= get_logger("silver.main")

def run_pipeline():
    log.info("Iniciando pipeline ETL - Capa Silver v2")
    
    # Paso 1: Obtener tags disponibles en Bronze
    tags = get_available_tags(RUTA_BRONZE)
    if not tags:
        log.error("No se encontraron tags en Bronze. Terminando proceso.")
        return
    tags_target = [t for t in RANGOS_NORMALES.keys() if t in tags]
    log.info("Cargando %d tags desde Bronze...", len(tags_target))


    datos_por_tag, resumen_df = cargar_datos_transformador_from_bronze(RUTA_BRONZE, tags_target)
    if not datos_por_tag:
        raise SystemExit("No se pudo cargar ningún tag desde Bronze")


    df = consolidar_datos_transformador(datos_por_tag)
    print(f" Datos consolidados: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")
    df = limpiar_nombres_columnas_transformador(df)
    df = convertir_tipos_transformador(df)


    stats_f = analizar_valores_faltantes_transformador(df)
    df_no_missing, missing_original, interpolated = tratar_valores_faltantes_transformador(df)


    criterios_tecnicos, criterios_combinados = definir_criterios_transformador()
    df_classified, estadisticas_estados = clasificar_estados_operacionales(df_no_missing, criterios_tecnicos, criterios_combinados)


    metricas = generar_reporte_calidad_transformador(df_classified, missing_original, interpolated, estadisticas_estados)


    resultados = guardar_dataset_transformador(
    df_classified,
    ruta_destino=RUTA_PROCESSED,
    metricas_calidad=metricas,
    criterios_tecnicos=criterios_tecnicos,
    silver_table_path=RUTA_SILVER,
    mode="overwrite",
    exportar_csv_parquet=False,
    )


    log.info("Pipeline Silver completado")
    return {
    "resumen_carga": resumen_df,
    "metricas_calidad": metricas,
    "resultados_guardado": resultados,
    }
    
if __name__ == "__main__":
    run_pipeline()