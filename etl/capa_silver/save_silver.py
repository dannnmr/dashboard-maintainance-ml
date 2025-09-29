from pathlib import Path
from datetime import datetime
from duckdb import df
import pandas as pd
from deltalake import write_deltalake


def guardar_dataset_transformador(
df: pd.DataFrame,
ruta_destino: Path,
metricas_calidad: dict,
criterios_tecnicos: dict,
silver_table_path: Path,
mode: str = "overwrite",
exportar_csv_parquet: bool = False,
):
    ruta_destino = Path(ruta_destino); ruta_destino.mkdir(parents=True, exist_ok=True)
    silver_table_path = Path(silver_table_path); silver_table_path.mkdir(parents=True, exist_ok=True)


# Preparar DataFrame para Delta
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in out.columns:
        raise ValueError("No se encontró columna 'timestamp' para Silver.")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["year"] = out["timestamp"].dt.year.astype("int32")
    out["month"] = out["timestamp"].dt.month.astype("int8")


# Escribir tabla Delta particionada
    write_deltalake(str(silver_table_path), out, mode=mode, partition_by=["year", "month"])


    resultados = {"silver_delta": {"exito": True, "path": str(silver_table_path), "mode": mode}}

    try:
        archivo_metadatos = ruta_destino / "transformador_data_metadatos_tecnicos.txt"
        with open(archivo_metadatos, "w", encoding="utf-8") as f:
            periodo_min = out["timestamp"].min(); periodo_max = out["timestamp"].max()
            f.write("METADATOS TÉCNICOS - TRANSFORMADOR ELÉCTRICO\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp de procesamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Tipo de equipo: Transformador de Potencia\n")
            f.write(f"Período de datos: {periodo_min} a {periodo_max}\n")
            f.write(f"Duración del monitoreo: {periodo_max - periodo_min}\n")
            f.write("Resolución temporal: 1 hora\n")
            f.write(f"Total de registros: {len(out):,}\n")
            f.write(f"Total de variables: {df.shape[1]}\n\n")
            f.write("CRITERIOS DE CLASIFICACIÓN OPERACIONAL\n")
            f.write("-" * 40 + "\n\n")
            for nombre, criterio in criterios_tecnicos.items():
                f.write(f"{nombre.upper()}:\n")
                f.write(f" Descripción: {criterio['descripcion']}\n")
                f.write(f" Normal: {criterio['normal']['min']} - {criterio['normal']['max']}\n")
                f.write(f" Alerta: {criterio['alerta']['min']} - {criterio['alerta']['max']}\n")
                f.write(f" Crítico: {criterio['critico']['min']} - {criterio['critico']['max']}\n\n")
            f.write("DISTRIBUCIÓN DE ESTADOS OPERACIONALES\n")
            f.write("-" * 40 + "\n\n")
            for estado, cantidad in metricas_calidad['estados_operacionales'].items():
                porcentaje = (cantidad / len(df) * 100) if len(df) > 0 else 0
                f.write(f"{estado}: {cantidad:,} registros ({porcentaje:.1f}%)\n")
            f.write(f"\nCompletitud de datos: {metricas_calidad['completitud_total']:.2f}%\n")
            f.write("\nVARIABLES POR CATEGORÍA TÉCNICA\n")
            f.write("-" * 40 + "\n\n")
            for categoria, cantidad in metricas_calidad['variables_por_categoria'].items():
                f.write(f"{categoria}: {cantidad} variables\n")
        resultados["metadatos_tecnicos"] = {"exito": True, "path": str(archivo_metadatos)}
    except Exception as e:
        resultados["metadatos_tecnicos"] = {"exito": False, "error": str(e)}
        
