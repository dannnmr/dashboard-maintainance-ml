# path: gold/features_electrical.py
import numpy as np
import pandas as pd

def crear_features_electricos_avanzados(df: pd.DataFrame, parametros: dict):
    print(" CREACIÓN DE FEATURES ELÉCTRICOS ")
    print("=" * 45)

    dfe = df.copy()
    feats = []

    # 1) Carga y utilización
    print(" Calculando factores de carga y utilización...")
    if "current_load_value" in df.columns and "power_apparent_value" in df.columns:
        cap_kva = parametros["capacidad_nominal_mva"] * 1000.0
        i_nom = cap_kva / (parametros["voltaje_nominal_kv"] * np.sqrt(3))
        dfe["load_factor_current"] = df["current_load_value"] / (i_nom + 1e-6)
        dfe["load_factor_power"]   = df["power_apparent_value"] / (cap_kva + 1e-6)
        dfe["overload_indicator"]  = (
            (dfe["load_factor_current"] > parametros["umbral_sobrecarga"]) |
            (dfe["load_factor_power"]   > parametros["umbral_sobrecarga"])
        ).astype(int)
        feats += ["load_factor_current","load_factor_power","overload_indicator"]
        print(" Factores de carga: 3 features")

    # 2) Eficiencia y pérdidas
    print("Analizando eficiencia y pérdidas eléctricas...")
    if all(v in df.columns for v in ["current_load_value","voltage_value","power_apparent_value"]):
        s_teor = (np.sqrt(3) * df["voltage_value"] * df["current_load_value"]) / 1000.0
        dfe["efficiency_indicator"]   = df["power_apparent_value"] / (s_teor + 1e-6)
        fp = 0.9
        p_est = df["power_apparent_value"] * fp
        dfe["power_factor_estimated"] = p_est / (df["power_apparent_value"] + 1e-6)
        dfe["relative_losses"]        = 1.0 - dfe["efficiency_indicator"]
        feats += ["efficiency_indicator","power_factor_estimated","relative_losses"]
        print(" Features de eficiencia: 3 features")

    # 3) OLTC
    print("Analizando comportamiento del sistema OLTC...")
    if "tap_position_value" in df.columns:
        tap_ch = df["tap_position_value"].diff().abs()
        for w in [24,168,720]:
            dname = w // 24
            dfe[f"tap_operations_{dname}d"] = tap_ch.rolling(window=w, min_periods=1).sum()
            feats.append(f"tap_operations_{dname}d")
        tap_c = 8.5
        dfe["tap_deviation_center"] = (df["tap_position_value"] - tap_c).abs()
        dfe["tap_extreme_position"] = ((df["tap_position_value"] <= 2) | (df["tap_position_value"] >= 15)).astype(int)
        dfe["tap_instability"]      = tap_ch.rolling(window=24, min_periods=1).std()
        feats += ["tap_deviation_center","tap_extreme_position","tap_instability"]
        print(f" Features OLTC: {len([f for f in feats if 'tap' in f])} features")

    # 4) Estabilidad eléctrica
    print("Creando indicadores de estabilidad eléctrica...")
    for var in ["current_load_value","voltage_value","power_apparent_value"]:
        if var in df.columns:
            mean24 = df[var].rolling(window=24, min_periods=6).mean()
            std24  = df[var].rolling(window=24, min_periods=6).std()
            dfe[f"{var}_stability"] = std24 / (mean24 + 1e-6)
            dfe[f"{var}_transient"] = ((df[var].diff().abs()) > (df[var].rolling(window=168).std() * 2)).astype(int)
            feats += [f"{var}_stability", f"{var}_transient"]
    print(f" Indicadores de estabilidad: {len([f for f in feats if 'stability' in f or 'transient' in f])} features")

    # 5) Combinaciones
    print(" Creando features de combinaciones eléctricas...")
    if all(v in df.columns for v in ["current_load_value","voltage_value"]) and \
       all(f in dfe.columns for f in ["load_factor_current","efficiency_indicator"]):
        dfe["electrical_stress_index"] = 0.6 * dfe["load_factor_current"] + 0.4 * (1.0 - dfe["efficiency_indicator"])
        dfe["suboptimal_operation"] = (
            (dfe["load_factor_current"] < 0.3) |
            (dfe["load_factor_current"] > 1.1) |
            (dfe["efficiency_indicator"] < 0.95)
        ).astype(int)
        feats += ["electrical_stress_index","suboptimal_operation"]
        print(" Combinaciones eléctricas: 2 features")

    # 6) Tendencias de degradación
    print("* Analizando tendencias de degradación eléctrica...")
    if "efficiency_indicator" in dfe.columns:
        w = 24 * 30
        dfe["efficiency_degradation_trend"] = (
            dfe["efficiency_indicator"].rolling(window=w,   min_periods=168).mean() -
            dfe["efficiency_indicator"].rolling(window=24, min_periods=12).mean()
        )
        dfe["efficiency_degradation_accel"] = dfe["efficiency_degradation_trend"].diff()
        feats += ["efficiency_degradation_trend","efficiency_degradation_accel"]
        print(" Tendencias de degradación: 2 features")

    print("* RESUMEN DE FEATURES ELÉCTRICOS:")
    print(f"- Total features eléctricos creados: {len(feats)}")
    print(f"- Dimensiones del dataset: {dfe.shape[0]:,} × {dfe.shape[1]}")

    muestra = feats[-10:] if len(feats) >= 10 else feats
    validos = 0
    for f in muestra:
        if f in dfe.columns:
            pct = (dfe[f].notna().sum() / len(dfe) * 100.0) if len(dfe) else 0.0
            if pct > 50:
                validos += 1
    print(f"-Features con >50% datos válidos: {validos}/{len(muestra)} (muestra)")

    return dfe, feats
