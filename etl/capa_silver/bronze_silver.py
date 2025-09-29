from pathlib import Path
import pandas as pd
import numpy as np
from deltalake import DeltaTable
from typing import Dict, List, Tuple
from logger_silver import get_logger


log = get_logger("silver.bronze")

def _schema_names(dt: DeltaTable) -> set:
    """Obtiene los nombres de columnas del esquema Delta de forma robusta."""
    try:
        sch = dt.schema()
        try:
            return set(getattr(sch, "names", []))
        except Exception:
            return {f.name for f in getattr(sch, "fields", [])}
    except Exception:
        return set()
    

def get_available_tags(bronze_path: Path) -> list[str]:
    """
    Devuelve la lista de tags disponibles en la tabla Bronze, evitando depender
    únicamente de dt.schema().names (que puede no reflejar bien las columnas).
    """
    if not bronze_path.exists():
        log.warning("Bronze path no existe: %s", bronze_path)
        return []

    try:
        dt = DeltaTable(str(bronze_path))

        # 1) Intento directo: pedir solo 'tag'
        try:
            pa = dt.to_pyarrow_table(columns=["tag"])
            tags = (
                pa.to_pandas()["tag"]
                .astype(str).dropna().unique().tolist()
            )
            if tags:
                log.info("Tags disponibles en Bronze: %d", len(tags))
                return tags
        except Exception as e:
            log.debug("No se pudo pedir solo 'tag' (%s). Intento leyendo todo.", e)

        # 2) Fallback: leer todo y buscar la columna 'tag'
        pdf = dt.to_pyarrow_table().to_pandas()
        cols = list(pdf.columns.astype(str))
        log.info("Columnas detectadas en Bronze: %s", cols)
        if "tag" not in cols:
            log.warning("La tabla Bronze no tiene columna 'tag' (según DataFrame).")
            return []

        tags = (
            pdf["tag"].astype(str)
            .dropna().unique().tolist()
        )
        log.info("Tags disponibles en Bronze: %d", len(tags))
        return tags

    except Exception as e:
        log.exception("Error al listar tags en Bronze: %s", e)
        return []

def cargar_tag_desde_bronze(tag: str, bronze_path: Path) -> pd.DataFrame | None:
    """
    Lee un 'tag' desde Bronze (Delta) y devuelve un DataFrame con:
      - 'timestamp' (UTC, limpio)
      - columna renombrada con el nombre del tag (a partir de value/value_text/value_bool)
    """
    try:
        if not bronze_path.exists():
            print(f"  Bronze path no existe: {bronze_path}")
            return None

        dt = DeltaTable(str(bronze_path))
        names = _schema_names(dt)

        # Solo pedimos columnas que EXISTEN en el Delta
        candidates = ["timestamp", "ts", "tag", "value", "value_text", "value_bool"]
        request_cols = [c for c in candidates if c in names]

        print(f"   Intentando cargar desde Delta: {bronze_path.resolve()}")
        print(f"   Esquema Delta (muestra): {sorted(list(names))[:10]} ...")
        print(f"   Columnas solicitadas (filtradas): {request_cols}")

        # Intentamos pushdown por columnas + filtro por tag
        try:
            if request_cols:
                pa_tbl = dt.to_pyarrow_table(columns=request_cols, filters=[("tag", "=", tag)])
            else:
                pa_tbl = dt.to_pyarrow_table(filters=[("tag", "=", tag)])
        except Exception as e:
            print(f"   Falló lectura con columnas especificadas ({e}). Leyendo todo y filtrando en pandas…")
            pa_tbl = dt.to_pyarrow_table()

        pdf = pa_tbl.to_pandas()

        # Filtro por tag (por si el filtro no se aplicó en el scan)
        if "tag" in pdf.columns:
            pdf = pdf[pdf["tag"].astype(str) == str(tag)].copy()

        if pdf.empty:
            print(f"   Sin filas para tag='{tag}'")
            return None

        # Normalizar timestamp (aceptamos 'timestamp' o 'ts' si existe)
        tscol = "timestamp" if "timestamp" in pdf.columns else ("ts" if "ts" in pdf.columns else None)
        if tscol:
            pdf[tscol] = pd.to_datetime(pdf[tscol], utc=True, errors="coerce")
            if tscol != "timestamp":
                pdf = pdf.rename(columns={tscol: "timestamp"})
        else:
            # No hay timestamp; no podemos continuar
            print("    No existe columna temporal ('timestamp' o 'ts') en la tabla.")
            return None

        # Limpiar timestamps inválidos
        registros_originales = len(pdf)
        pdf = pdf.dropna(subset=["timestamp"]).copy()
        if len(pdf) < registros_originales:
            print(f"   Eliminadas {registros_originales - len(pdf):,} filas con timestamp inválido.")

        if pdf.empty:
            print(f"   No quedan datos válidos para tag='{tag}' tras limpiar timestamps.")
            return None

        # Unificar la columna de valor: prioridad value > value_text (num) > value_bool
        if "value" not in pdf.columns:
            pdf["value"] = np.nan

        if pdf["value"].notna().sum() == 0 and "value_text" in pdf.columns:
            maybe_num = pd.to_numeric(pdf["value_text"], errors="coerce")
            if maybe_num.notna().sum() > 0:
                pdf["value"] = maybe_num

        if pdf["value"].notna().sum() == 0 and "value_bool" in pdf.columns:
            pdf["value"] = pdf["value_bool"].astype("Int64")

        # Mantener solo timestamp + valor renombrado al tag
        pdf = pdf[["timestamp", "value"]].rename(columns={"value": tag})
        pdf = pdf.sort_values("timestamp").reset_index(drop=True)

        # Estadística rápida
        vals_num = pd.to_numeric(pdf[tag], errors="coerce")
        vmin, vmax, vmean = vals_num.min(skipna=True), vals_num.max(skipna=True), vals_num.mean(skipna=True)
        fecha_ini, fecha_fin = pdf["timestamp"].min(), pdf["timestamp"].max()
        print(f"   {len(pdf):,} registros | Rango: {fecha_ini:%Y-%m-%d} a {fecha_fin:%Y-%m-%d}")
        if pd.notna(vmin) and pd.notna(vmax) and pd.notna(vmean):
            print(f"   Valores: {vmin:.2f} a {vmax:.2f} | Promedio: {vmean:.2f}")

        return pdf

    except Exception as e:
        print(f"   Error al cargar tag='{tag}': {e}")
        return None

def cargar_datos_transformador_from_bronze(
	bronze_path: Path, tags_a_cargar: List[str]
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
	datos_por_tag: Dict[str, pd.DataFrame] = {}
	resumen: List[dict] = []
	for tag in tags_a_cargar:
		df = cargar_tag_desde_bronze(tag, bronze_path)
		if df is None or df.empty:
			continue
		datos_por_tag[tag] = df
		vals = pd.to_numeric(df[tag], errors="coerce")
		resumen.append({
			"variable": tag,
			"registros": len(df),
			"fecha_inicio": df["timestamp"].min(),
			"fecha_fin": df["timestamp"].max(),
			"duracion_dias": (df["timestamp"].max() - df["timestamp"].min()).days if len(df) else 0,
			"valores_nulos": vals.isna().sum(),
			"valor_min": vals.min(skipna=True),
			"valor_max": vals.max(skipna=True),
			"valor_medio": vals.mean(skipna=True),
		})
	resumen_df = (
		pd.DataFrame(resumen).sort_values("variable").reset_index(drop=True) if resumen else pd.DataFrame()
	)
	return datos_por_tag, resumen_df