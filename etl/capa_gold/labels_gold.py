# path: gold/labels_gold.py
import numpy as np
import pandas as pd

def crear_etiquetas_prediccion_transformador(df_principal: pd.DataFrame,
                                             df_anomalias: pd.DataFrame | None,
                                             parametros: dict):
    print("    CREACIÓN DE ETIQUETAS DE PREDICCIÓN")
    print("=" * 40)

    df = df_principal.copy()
    H_dias  = parametros["horizonte_prediccion_dias"]
    H_horas = H_dias * 24
    print(f"    Horizonte de predicción: {H_dias} días ({H_horas} horas)")

    # Asegurar índice temporal
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.sort_index()

    # Etiquetas base
    df["falla_30d"] = 0
    df["estado_futuro"] = "NORMAL"
    df["rul_dias"] = float(H_dias)
    df["severidad_futura"] = 0.0

    # 1) Eventos críticos
    print("   Identificando eventos críticos...")
    eventos = []
    if df_anomalias is not None and len(df_anomalias) > 0:
        an = df_anomalias.copy()
        if "timestamp" in an.columns:
            an["timestamp"] = pd.to_datetime(an["timestamp"], utc=True, errors="coerce")
            an = an.sort_values("timestamp").set_index("timestamp")
        elif not isinstance(an.index, pd.DatetimeIndex):
            an.index = pd.to_datetime(an.index, utc=True, errors="coerce")
            an = an.sort_index()

        if "estado_operacional" in an.columns:
            eventos = an[an["estado_operacional"] == "CRITICO"].index.tolist()
        elif "nivel_severidad" in an.columns:
            eventos = an[an["nivel_severidad"] >= 2].index.tolist()

    if not eventos:
        print(" No se encontraron marcadores de eventos críticos")
    else:
        print(f"    Eventos críticos identificados: {len(eventos)}")
        print(f"    Período de eventos: {min(eventos)} a {max(eventos)}")

    # 2) Binaria + RUL
    print("    Creando etiquetas binarias de predicción...")
    n_eventos, n_pos = 0, 0
    for ev in eventos:
        ini, fin = ev - pd.Timedelta(hours=H_horas), ev
        mask = (df.index >= ini) & (df.index < fin)
        idxs = df.index[mask]
        if len(idxs) > 0:
            df.loc[idxs, "falla_30d"] = 1
            tdiff_h = (ev - idxs).total_seconds() / 3600.0
            df.loc[idxs, "rul_dias"] = np.maximum(0.0, tdiff_h / 24.0)
            n_eventos += 1
            n_pos += len(idxs)
    print(f"    Eventos procesados: {n_eventos}")
    print(f"    Registros marcados como positivos: {n_pos:,}")

    # 3) Multiclase por RUL
    print("   Creando etiquetas multi-clase...")
    u_crit, u_alert = 7, 15
    m_crit  = df["rul_dias"] <= u_crit
    m_alert = (df["rul_dias"] > u_crit) & (df["rul_dias"] <= u_alert)
    df.loc[m_crit,  "estado_futuro"] = "CRITICO"
    df.loc[m_alert, "estado_futuro"] = "ALERTA"

    dist = df["estado_futuro"].value_counts()
    print("   Distribución de estados futuros:")
    for estado, cantidad in dist.items():
        pct = (cantidad / len(df) * 100.0) if len(df) else 0.0
        emoji = {"NORMAL":"✅","ALERTA":"⚠️","CRITICO":"🚨"}.get(estado,"📊")
        print(f"      {emoji} {estado}: {cantidad:,} ({pct:.1f}%)")

    # 4) Severidad progresiva
    print("    Creando etiquetas de severidad progresiva...")
    df["severidad_futura"] = (100.0 * (1.0 - df["rul_dias"] / float(H_dias))).clip(0, 100)

    # 5) Extras
    print("    Creando features adicionales de etiquetado...")
    df["dias_proximo_evento"] = float(H_dias)
    if eventos:
        eventos = sorted(eventos)
        for i, ts in enumerate(df.index):
            fut = [e for e in eventos if e > ts]
            if fut:
                prox = fut[0]
                dias = (prox - ts).total_seconds() / (24 * 3600.0)
                df.iloc[i, df.columns.get_loc("dias_proximo_evento")] = min(dias, H_dias)

    df["proximidad_evento"] = np.exp(-df["dias_proximo_evento"] / 10.0)
    df["riesgo_acumulativo"] = 0.0
    for ev in eventos:
        diff_h = (np.abs(df.index - ev).total_seconds()) / 3600.0
        infl = np.exp(-(diff_h) / (H_horas / 2.0))
        df["riesgo_acumulativo"] += infl
    max_r = df["riesgo_acumulativo"].max()
    if pd.notna(max_r) and max_r > 0:
        df["riesgo_acumulativo"] = df["riesgo_acumulativo"] / max_r

    # 6) Validación de etiquetas
    print("   Validando calidad de etiquetas...")
    tasa = float(df["falla_30d"].mean()) if len(df) else 0.0
    print(f"   - Tasa de casos positivos: {tasa:.3f} ({tasa*100:.1f}%)")
    if tasa < 0.01:
        print("    Advertencia: Muy pocos casos positivos (<1%)")
    elif tasa > 0.5:
        print("   Advertencia: Demasiados casos positivos (>50%)")
    else:
        print("   Balance de clases apropiado para mantenimiento predictivo")

    rul_ok = ((df["rul_dias"] >= 0) & (df["rul_dias"] <= H_dias)).mean() if len(df) else 1.0
    print(f"   - RUL válido: {rul_ok*100:.1f}% de registros")
    sev_mean = float(df["severidad_futura"].mean()) if len(df) else 0.0
    sev_max  = float(df["severidad_futura"].max())  if len(df) else 0.0
    print(f"   - Severidad media: {sev_mean:.1f}%, máxima: {sev_max:.1f}%")

    etiquetas = [
        "falla_30d","estado_futuro","rul_dias","severidad_futura",
        "dias_proximo_evento","proximidad_evento","riesgo_acumulativo"
    ]
    print("   * RESUMEN DE ETIQUETADO:")
    print(f"   - Etiquetas creadas: {len(etiquetas)}")
    print(f"   - Dataset etiquetado: {df.shape[0]:,} × {df.shape[1]}")
    print("    - Listo para entrenamiento de modelos predictivos")

    return df, etiquetas
