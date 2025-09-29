# path: silver/classify.py
from duckdb import df
import pandas as pd
from pyparsing import col
from logger_silver import get_logger


log = get_logger("silver.classify_silver")


def clasificar_estados_operacionales(df: pd.DataFrame, criterios_tecnicos: dict, criterios_combinados: dict):
    log.info("Clasificando estados operacionales...")
    out = df.copy()
    out["estado_operacional"] = "NORMAL"
    out["nivel_severidad"] = 0
    out["variables_anomalas"] = ""
    out["descripcion_anomalia"] = ""


# individuales
    for nombre, crit in criterios_tecnicos.items():
        cols = [c for c in out.columns if all(t in c.lower() for t in nombre.split('_'))]
        if not cols:
            continue
        for col in cols:
            if not pd.api.types.is_numeric_dtype(out[col]):
                continue
            v = out[col]
            mask_c = (v >= crit['critico']['min']) & (v < crit['critico']['max'])
            mask_a = (v >= crit['alerta']['min']) & (v < crit['alerta']['max'])
            idx_c = out.index[mask_c]
            out.loc[idx_c, 'estado_operacional'] = 'CRITICO'
            out.loc[idx_c, 'nivel_severidad'] = 2
            for i in idx_c:
                out.at[i, 'variables_anomalas'] = (out.at[i, 'variables_anomalas'] + ", " if out.at[i, 'variables_anomalas'] else "") + col
                desc = f"{crit['descripcion']} CRÍTICO"
                out.at[i, 'descripcion_anomalia'] = (out.at[i, 'descripcion_anomalia'] + "; " if out.at[i, 'descripcion_anomalia'] else "") + desc
            idx_a = out.index[mask_a & (out['nivel_severidad'] < 2)]
            out.loc[idx_a, 'estado_operacional'] = 'ALERTA'
            out.loc[idx_a, 'nivel_severidad'] = 1
            for i in idx_a:
                out.at[i, 'variables_anomalas'] = (out.at[i, 'variables_anomalas'] + ", " if out.at[i, 'variables_anomalas'] else "") + col
                desc = f"{crit['descripcion']} ALERTA"
                out.at[i, 'descripcion_anomalia'] = (out.at[i, 'descripcion_anomalia'] + "; " if out.at[i, 'descripcion_anomalia'] else "") + desc


# combinados: sobrecarga térmica
    cols_i = [c for c in out.columns if 'current' in c.lower()]
    cols_o = [c for c in out.columns if ('oil' in c.lower() and 'temp' in c.lower())]
    if cols_i and cols_o:
        for ci in cols_i:
            for co in cols_o:
                mask = (out[ci] > 2000) & (out[co] > 70)
                idx = out.index[mask]
                out.loc[idx, 'estado_operacional'] = 'CRITICO'
                out.loc[idx, 'nivel_severidad'] = 2
                for i in idx:
                    desc = "SOBRECARGA TÉRMICA COMBINADA"
                    if desc not in out.at[i, 'descripcion_anomalia']:
                        out.at[i, 'descripcion_anomalia'] = (out.at[i, 'descripcion_anomalia'] + "; " if out.at[i, 'descripcion_anomalia'] else "") + desc


    # combinados: estrés térmico
    cols_h = [c for c in out.columns if 'hot' in c.lower()]
    if cols_h and cols_o:
        for ch in cols_h:
            for co in cols_o:
                diff = out[ch] - out[co]
                mask = (diff > 25) & (out['nivel_severidad'] < 2)
                idx = out.index[mask]
                out.loc[idx, 'estado_operacional'] = 'ALERTA'
                out.loc[idx, 'nivel_severidad'] = 1
                for i in idx:
                    desc = f"GRADIENTE TÉRMICO EXCESIVO ({diff.at[i]:.1f}°C)"
                    if "GRADIENTE" not in out.at[i, 'descripcion_anomalia']:
                        out.at[i, 'descripcion_anomalia'] = (out.at[i, 'descripcion_anomalia'] + "; " if out.at[i, 'descripcion_anomalia'] else "") + desc


    cont = out['estado_operacional'].value_counts().to_dict()
    return out, cont