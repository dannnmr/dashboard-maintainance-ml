# path: gold/features_thermal.py
import pandas as pd

def crear_features_termicos_avanzados(df: pd.DataFrame, parametros: dict):
    print("CREACIÓN DE FEATURES TÉRMICOS AVANZADOS")
    print("=" * 45)

    df_thermal = df.copy()
    feats = []

    #  Calculo de gradientes térmicos
    print("\n Calculando gradientes térmicos fundamentales...")
    if "temp_spot_hot_value" in df.columns and "temp_oil_value" in df.columns:
        df_thermal["gradient_hot_oil"] = df["temp_spot_hot_value"] - df["temp_oil_value"]
        feats.append("gradient_hot_oil")
        if "temp_ambient_value" in df.columns:
            df_thermal["gradient_normalized"] = (
                (df["temp_spot_hot_value"] - df["temp_oil_value"]) /
                (df["temp_oil_value"] - df["temp_ambient_value"] + 1e-6)
            )
            df_thermal["temp_rise_hot"] = df["temp_spot_hot_value"] - df["temp_ambient_value"]
            df_thermal["temp_rise_oil"] = df["temp_oil_value"] - df["temp_ambient_value"]
            feats += ["gradient_normalized", "temp_rise_hot", "temp_rise_oil"]
        print(f"    Gradientes térmicos: {len([f for f in feats if 'gradient' in f or 'rise' in f])} features")

    # 2) Inercia térmica
    print("\nAnalizando inercia térmica y velocidades de cambio...")
    vars_t = ["temp_oil_value","temp_spot_hot_value","temp_ambient_value","temp_oil_oltc_value","temp_bubbling_value"]
    for v in vars_t:
        if v in df.columns:
            df_thermal[f"{v}_rate"] = df[v].diff()
            df_thermal[f"{v}_accel"] = df_thermal[f"{v}_rate"].diff()
            df_thermal[f"{v}_rate_smooth"] = df_thermal[f"{v}_rate"].rolling(window=6, min_periods=1).mean()
            feats += [f"{v}_rate", f"{v}_accel", f"{v}_rate_smooth"]
    print(f"    Features de inercia térmica: {len([f for f in feats if 'rate' in f or 'accel' in f])} features")

    # 3) Eficiencia térmica y carga
    print("\n Calculando eficiencia térmica y relaciones con carga eléctrica...")
    if all(v in df.columns for v in ["temp_oil_value","current_load_value","temp_ambient_value"]):
        df_thermal["thermal_efficiency"] = (df["temp_oil_value"] - df["temp_ambient_value"]) / (df["current_load_value"] + 1e-6)
        df_thermal["thermal_loading_factor"] = (
            (df["temp_oil_value"] - df["temp_ambient_value"]) /
            (parametros["temp_aceite_nominal"] - parametros["temp_ambiente_nominal"])
        )
        feats += ["thermal_efficiency","thermal_loading_factor"]
        print("    Features de eficiencia térmica: 2 features")

    # 4) Ciclos térmicos (opcional, usando find_peaks)
    print("\n Detectando ciclos térmicos y patrones de fatiga...")
    if "temp_oil_value" in df.columns:
        try:
            from scipy.signal import find_peaks  # opcional
            s = df["temp_oil_value"].dropna()
            if len(s) > 48:
                peaks, _ = find_peaks(s.values, height=s.mean(), distance=12)
                ventana = 24 * 7
                df_thermal["thermal_cycles_7d"] = 0.0
                n = len(df_thermal)
                for i in range(n):
                    ini, fin = max(0, i - ventana), i + 1
                    p = len([px for px in peaks if ini <= px < fin])
                    df_thermal.iloc[i, df_thermal.columns.get_loc("thermal_cycles_7d")] = float(p)
                feats.append("thermal_cycles_7d")
                print("   Análisis de ciclos térmicos: 1 feature")
        except Exception as e:
            print(f"    Saltando análisis de ciclos (scipy no disponible o error): {e}")

    # 5) Estrés térmico combinado
    print("\nCreando indicadores de estrés térmico combinado...")
    if all(v in df.columns for v in ["temp_spot_hot_value","temp_oil_value"]):
        if "gradient_hot_oil" not in df_thermal.columns:
            df_thermal["gradient_hot_oil"] = df["temp_spot_hot_value"] - df["temp_oil_value"]
            feats.append("gradient_hot_oil")
        temp_max_nominal = 100
        df_thermal["thermal_stress_index"] = (
            0.7 * (df["temp_spot_hot_value"] / temp_max_nominal) +
            0.3 * (df_thermal["gradient_hot_oil"] / parametros["umbral_gradiente_termico"])
        )
        ventana = 24 * 30
        df_thermal["overheating_trend"] = (
            df["temp_oil_value"].rolling(window=ventana, min_periods=168).mean() -
            df["temp_oil_value"].rolling(window=24, min_periods=12).mean()
        )
        feats += ["thermal_stress_index","overheating_trend"]
        print("   Indicadores de estrés térmico: 2 features")

    # 6) Temperatura de burbujeo (gases)
    print("\n💨 Analizando temperatura de burbujeo como indicador de gases...")
    if "temp_bubbling_value" in df.columns:
        mean7 = df["temp_bubbling_value"].rolling(window=24*7,  min_periods=24).mean()
        std7  = df["temp_bubbling_value"].rolling(window=24*7,  min_periods=24).std()
        df_thermal["bubbling_anomaly"] = (df["temp_bubbling_value"] - mean7) / (std7 + 1e-6)
        df_thermal["gas_formation_trend"] = (
            df["temp_bubbling_value"].rolling(window=24*30, min_periods=168).mean() -
            df["temp_bubbling_value"].rolling(window=24*7,  min_periods=24).mean()
        )
        feats += ["bubbling_anomaly","gas_formation_trend"]
        print("   Features de análisis de gases: 2 features")

    # Resumen
    print("\n RESUMEN DE FEATURES TÉRMICOS:")
    print(f"    Total features térmicos creados: {len(feats)}")
    print(f"    Dimensiones del dataset: {df_thermal.shape[0]:,} × {df_thermal.shape[1]}")

    # Chequeo rápido de completitud
    muestra = feats[-10:] if len(feats) >= 10 else feats
    validos = 0
    for f in muestra:
        if f in df_thermal.columns:
            pct = (df_thermal[f].notna().sum() / len(df_thermal) * 100.0) if len(df_thermal) else 0.0
            if pct > 50:
                validos += 1
    print(f"   Features con >50% datos válidos: {validos}/{len(muestra)} (muestra)")

    return df_thermal, feats
