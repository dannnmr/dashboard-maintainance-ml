# path: gold/validate_gold.py
import pandas as pd

def validar_coherencia_tecnica_transformador(df: pd.DataFrame, parametros: dict) -> dict:
    print(" VALIDACIÓN TÉCNICA DE COHERENCIA FÍSICA")
    print("=" * 45)

    validaciones, alertas = {}, []

    # 1) Coherencia térmica
    if "temp_spot_hot_value" in df.columns and "temp_oil_value" in df.columns:
        th = df["temp_spot_hot_value"].dropna()
        to = df["temp_oil_value"].dropna()
        idx = th.index.intersection(to.index)
        if len(idx) > 0:
            coh = (th.loc[idx] >= to.loc[idx]).mean()
            validaciones["coherencia_termica"] = coh * 100
            print(f" Coherencia térmica: {coh*100:.1f}%")
            if coh < 0.95:
                alertas.append(f" Coherencia térmica baja: {coh*100:.1f}%")
            grad = th.loc[idx] - to.loc[idx]
            validaciones["gradiente_termico_medio"] = grad.mean()
            validaciones["gradiente_termico_max"]   = grad.max()
            print(f"    Gradiente térmico medio: {grad.mean():.2f}°C")
            print(f"    Gradiente térmico máximo: {grad.max():.2f}°C")
            if grad.max() > parametros["umbral_gradiente_termico"]:
                alertas.append(f" Gradiente térmico excesivo: {grad.max():.1f}°C")

    # 2) Consistencia eléctrica
    if "current_load_value" in df.columns and "power_apparent_value" in df.columns:
        i = df["current_load_value"].dropna()
        s = df["power_apparent_value"].dropna()
        idx = i.index.intersection(s.index)
        if len(idx) > 0:
            i2, s2 = i.loc[idx], s.loc[idx]
            mask = (i2 > 10) & (s2 > 1)
            if mask.sum() > 0:
                ratio = s2[mask] / i2[mask]
                validaciones["ratio_potencia_corriente"] = ratio.mean()
                validaciones["variabilidad_ratio"] = (ratio.std() / ratio.mean()) if ratio.mean() > 0 else 0
                print(f" Ratio P/I medio: {ratio.mean():.3f} kVA/A")
                print(f"    Coeficiente de variación: {(ratio.std()/ratio.mean()*100):.1f}%")

    # 3) Rangos operacionales
    rangos = {
        "temp_oil_value": (0, 120),
        "temp_spot_hot_value": (0, 150),
        "temp_ambient_value": (-40, 60),
        "current_load_value": (0, 5000),
        "power_apparent_value": (0, 200),
        "voltage_value": (100, 150),
        "tap_position_value": (0, 20),
    }
    print("\nValidación de rangos operacionales:")
    for var, (mn, mx) in rangos.items():
        if var in df.columns:
            s = df[var].dropna()
            if len(s) > 0:
                fuera = ((s < mn) | (s > mx)).sum()
                pct_ok = (len(s) - fuera) * 100.0 / len(s)
                validaciones[f"validez_{var}"] = pct_ok
                emoji = "✅" if pct_ok > 95 else "⚠️" if pct_ok > 90 else "❌"
                print(f"   {emoji} {var}: {pct_ok:.1f}% válido")
                if pct_ok < 90:
                    alertas.append(f" {var}: {100 - pct_ok:.1f}% fuera de rango")

    # 4) Continuidad temporal (1H)
    if isinstance(df.index, pd.DatetimeIndex):
        diff = df.index.to_series().diff()
        gaps = diff[diff > pd.Timedelta(hours=1) * 1.5]
        n_gaps = len(gaps)
        validaciones["continuidad_temporal"] = (len(df) - n_gaps) * 100.0 / len(df) if len(df) else 100.0
        validaciones["numero_gaps"] = n_gaps
        print("\n* Continuidad temporal:")
        print(f"   - Gaps temporales detectados: {n_gaps}")
        print(f"   - Continuidad: {validaciones['continuidad_temporal']:.1f}%")
        if n_gaps > len(df) * 0.05:
            alertas.append(f" Muchos gaps temporales: {n_gaps} ({n_gaps/len(df)*100:.1f}%)")

    print("\n RESUMEN DE VALIDACIÓN TÉCNICA:")
    if not alertas:
        print("    Todos los criterios de validación técnica cumplidos")
        validaciones["estado_validacion"] = "EXITOSO"
    else:
        print(f"   Se detectaron {len(alertas)} alertas:")
        for a in alertas:
            print(f"      {a}")
        validaciones["estado_validacion"] = "ADVERTENCIAS"

    validaciones["alertas"] = alertas
    return validaciones
