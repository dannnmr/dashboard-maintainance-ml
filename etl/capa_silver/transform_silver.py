# path: silver/transform.py
import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from webcolors import names
from logger_silver import get_logger
from config_silver_v2 import CONSOLIDACION_MAX_DIAS

log = get_logger("silver.transform")

def consolidar_datos_transformador(datos_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
	if not datos_dict:
		raise ValueError("No se encontraron datos válidos para consolidar")
	log.info("Consolidando datos con alineación temporal (1H)...")

	indices = [df["timestamp"] for df in datos_dict.values() if "timestamp" in df.columns]
	if not indices:
		# concatenación simple por columnas, asumiendo que ya vienen alineados
		base = None
		for name, df in datos_dict.items():
			if base is None:
				base = df.copy()
			else:
				base = base.merge(df, how="outer", on="timestamp")
		base = base.set_index("timestamp").sort_index()
		return base

	inicio = max(s.min() for s in indices)
	fin = min(s.max() for s in indices)
	inicio = inicio.round("H")
	fin = fin.round("H")

	# Límite configurable desde config_silverv2.CONOSOLIDACION_MAX_DIAS
	if isinstance(CONSOLIDACION_MAX_DIAS, (int, float)) and CONSOLIDACION_MAX_DIAS > 0:
		max_span = pd.Timedelta(days=float(CONSOLIDACION_MAX_DIAS))
		if (fin - inicio) > max_span:
			fin = inicio + max_span
			log.info("Limitando rango temporal a %s días (config)", CONSOLIDACION_MAX_DIAS)
	else:
		log.info("Usando rango temporal completo: %s → %s (%s)", inicio, fin, fin - inicio)


	idx = pd.date_range(inicio, fin, freq="H")
	out = pd.DataFrame(index=idx)

	termicas = {
		"temperatura_aceite",
		"temperatura_punto_caliente",
		"temperatura_ambiente",
		"temperatura_burbujeo",
		"temperatura_aceite_OLTC",
	}
	electricas = {"corriente_carga", "voltaje", "potencia_aparente"}
	mecanicas = {"tap_position"}

	for nombre, df in datos_dict.items():
		if "timestamp" not in df.columns:
			log.warning("Saltando %s: sin timestamp", nombre)
			continue
		df = df[(df["timestamp"] >= inicio) & (df["timestamp"] <= fin)].copy()
		if df.empty:
			log.warning("%s sin datos en rango común", nombre)
			continue
		df = df.set_index("timestamp")
		# Sólo una columna de valor por tag
		val_col = [c for c in df.columns if c != "timestamp"]
		if not val_col:
			log.warning("%s sin columna de valor", nombre)
			continue
		col = val_col[0]
		if nombre in termicas:
			res = df[col].resample("H").mean().interpolate(method="time")
		elif nombre in electricas:
			res = df[col].resample("H").mean()
		elif nombre in mecanicas:
			res = df[col].resample("H").last().ffill()
		else:
			res = df[col].resample("H").mean().interpolate(method="linear")
		out[nombre + "_value"] = res.reindex(idx)
	return out



def limpiar_nombres_columnas_transformador(df: pd.DataFrame) -> pd.DataFrame:
	log.info("Limpiando nombres de columnas...")
	mapping = {
		'corriente_carga_value': 'current_load_value',
		'potencia_aparente_value': 'power_apparent_value',
		'tap_position_value': 'tap_position_value',
		'temperatura_aceite_value': 'temp_oil_value',
		'temperatura_aceite_OLTC_value': 'temp_oil_oltc_value',
		'temperatura_ambiente_value': 'temp_ambient_value',
		'temperatura_burbujeo_value': 'temp_bubbling_value',
		'temperatura_punto_caliente_value': 'temp_spot_hot_value',
		'voltaje_value': 'voltage_value',
	}
	cols = []
	for c in df.columns:
		if c in mapping:
			cols.append(mapping[c])
			continue
		name = str(c).lower()
		rep = {
			'temperatura': 'temp', 
			'corriente': 'current',
			'voltaje': 'voltage', 
			'potencia': 'power',
			'aparente': 'apparent', 
			'aceite': 'oil', 
			'ambiente': 'ambient', 
			'caliente': 'hot',
			'punto': 'spot', 
			'burbujeo': 'bubbling', 
			'posicion': 'position', 
			'carga': 'load'
		}
		for k, v in rep.items():
			name = name.replace(k, v)
		# Normalizar caracteres
		name = re.sub(r'[áéíóúñ]', lambda m: 'aeioun'['áéíóúñ'.index(m.group())], name)
		name = re.sub(r"\s+", "_", name)
		name = re.sub(r"[^a-z0-9_]", "", name)
		name = re.sub(r"_+", "_", name).strip("_") or "var_transformador"
		cols.append(name)
	df = df.copy()
	df.columns = _dedupe(cols)
	return df


def _dedupe(names):
    seen = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def convertir_tipos_transformador(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Convirtiendo tipos y validando rangos técnicos...")
    df = df.copy()
    rangos = {
    'temp': (-50, 200), 'oil': (0, 150), 'hot': (0, 180), 'ambient': (-40, 60),
    'current': (0, 5000), 'voltage': (0, 50000), 'power': (0, 100000),
    'position': (0, 20), 'tap': (0, 20)
    }
    for col in df.columns:
        s = pd.to_numeric(df[col], errors='coerce')
        applied = None
        lc = col.lower()
        for k, (mn, mx) in rangos.items():
            if k in lc:
                s[(s < mn) | (s > mx)] = np.nan
                applied = (mn, mx)
                break
        df[col] = s
    return df


def analizar_valores_faltantes_transformador(df: pd.DataFrame) -> dict:
    log.info("Analizando valores faltantes por categoría...")
    cats = {
    'Térmica': ['temp', 'oil', 'hot', 'ambient', 'bubbling'],
    'Eléctrica': ['current', 'voltage', 'power', 'apparent'],
    'Mecánica': ['position', 'tap'],
    'Otras': []
    }
    stats = {}
    for cat, terms in cats.items():
        cols = [c for c in df.columns if any(t in c.lower() for t in terms)] if terms else []
        if cat == 'Otras':
            tagged = {c for c in df.columns for ts in cats.values() if ts for t in ts if t in c.lower()}
            cols = [c for c in df.columns if c not in tagged]
        if not cols:
            continue
        missing = df[cols].isna().sum().sum()
        total = df[cols].size
        stats[cat] = {"columnas": len(cols), "valores_faltantes": int(missing), "porcentaje": (missing/total*100) if total else 0.0}
    # resumen global
    stats['__global__'] = {
    'total_valores': int(df.size),
    'total_missing': int(df.isna().sum().sum()),
    'porcentaje_total': float(df.isna().sum().sum() / df.size * 100) if df.size else 0.0,
    'columnas_afectadas': int((df.isna().sum() > 0).sum()),
    'total_columnas': int(len(df.columns))
    }
    return stats

def tratar_valores_faltantes_transformador(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    log.info("Interpolando valores faltantes por naturaleza de variable...")
    df2 = df.copy()
    before = int(df2.isna().sum().sum())
    if before == 0:
        log.info("No hay valores faltantes")
        return df2, 0, 0


    cats = {
    'Térmica': ['temp', 'oil', 'hot', 'ambient', 'bubbling'],
    'Eléctrica': ['current', 'voltage', 'power', 'apparent'],
    'Mecánica': ['position', 'tap']
    }
    # térmicas: spline si hay índice temporal, si no lineal
    if isinstance(df2.index, pd.DatetimeIndex):
        for col in [c for c in df2.columns if any(t in c.lower() for t in cats['Térmica'])]:
            if df2[col].isna().any():
                try:
                    df2[col] = df2[col].interpolate(method='spline', order=2, limit_direction='both')
                except Exception:
                    df2[col] = df2[col].interpolate(method='time', limit_direction='both')
        for col in [c for c in df2.columns if any(t in c.lower() for t in cats['Eléctrica'])]:
            if df2[col].isna().any():
                df2[col] = df2[col].interpolate(method='time', limit_direction='both')
    else:
        for col in [c for c in df2.columns if any(t in c.lower() for t in cats['Térmica'] + cats['Eléctrica'])]:
            if df2[col].isna().any():
                df2[col] = df2[col].interpolate(method='linear', limit_direction='both')
    # mecánicas: ffill/bfill
    mec_cols = [c for c in df2.columns if any(t in c.lower() for t in cats['Mecánica'])]
    if mec_cols:
        df2[mec_cols] = df2[mec_cols].ffill().bfill()


    after = int(df2.isna().sum().sum())
    interpol = before - after
    log.info("Interpolados: %d (faltantes: %d -> %d)", interpol, before, after)
    return df2, before, interpol