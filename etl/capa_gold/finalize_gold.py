# path: gold/finalize_gold.py
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from paths_gold import RUTA_GOLD_COMPLETE, RUTA_GOLD_TRAIN, RUTA_GOLD_VALID

def _try_import_write_deltalake():
    try:
        from deltalake import write_deltalake  # type: ignore
        return write_deltalake
    except Exception:
        return None

def finalizar_dataset_transformador(
    df_final: pd.DataFrame,
    features_termicos: list[str],
    features_electricos: list[str],
    etiquetas_creadas: list[str],
    ruta_features: Path,
    parametros: dict,
    save_parquet_csv: bool = True,
    save_delta: bool = True,
) -> dict:
    print("* FINALIZACIÓN Y GUARDADO DEL DATASET*")
    print("=" * 42)

    # 1) Resumen de features
    print(" Análisis completo de features creados...")
    variables_originales = [
        "current_load_value","power_apparent_value","tap_position_value",
        "temp_oil_value","temp_oil_oltc_value","temp_ambient_value",
        "temp_bubbling_value","temp_spot_hot_value","voltage_value",
        "estado_operacional","nivel_severidad","variables_anomalas","descripcion_anomalia",
    ]
    cat = {
        "Variables Originales": [c for c in df_final.columns if c in variables_originales],
        "Features Térmicos": features_termicos,
        "Features Eléctricos": features_electricos,
        "Etiquetas de Predicción": etiquetas_creadas,
        "Otros Features": []
    }
    todos = []
    for lst in cat.values():
        todos.extend(lst)
    cat["Otros Features"] = [c for c in df_final.columns if c not in todos]

    print(f"   Dimensiones del dataset final: {df_final.shape[0]:,} × {df_final.shape[1]}")
    print("   Distribución de features por categoría:")
    total_features = 0
    for k, v in cat.items():
        if len(v) > 0:
            print(f"      🔧 {k}: {len(v)} features")
            total_features += len(v)
    print(f"   * Total features: {total_features}")

    # 2) Optimización de memoria
    print("   🚀 Optimizando memoria y tipos de datos...")
    mem_ini = df_final.memory_usage(deep=True).sum() / 1024**2
    df_opt = df_final.copy()
    for c in df_opt.columns:
        if df_opt[c].dtype == "float64":
            if df_opt[c].notna().any():
                mn, mx = df_opt[c].min(), df_opt[c].max()
                if np.isfinite(mn) and np.isfinite(mx) and abs(mn) < 1e37 and abs(mx) < 1e37:
                    df_opt[c] = df_opt[c].astype("float32")
        elif df_opt[c].dtype == "int64":
            if df_opt[c].notna().any():
                mn, mx = df_opt[c].min(), df_opt[c].max()
                if mn >= 0 and mx <= 255:
                    df_opt[c] = df_opt[c].astype("uint8")
                elif mn >= -128 and mx <= 127:
                    df_opt[c] = df_opt[c].astype("int8")
                elif mn >= 0 and mx <= 65535:
                    df_opt[c] = df_opt[c].astype("uint16")
                elif mn >= -32768 and mx <= 32767:
                    df_opt[c] = df_opt[c].astype("int16")
                else:
                    df_opt[c] = df_opt[c].astype("int32")
    mem_fin = df_opt.memory_usage(deep=True).sum() / 1024**2
    red_pct = (mem_ini - mem_fin) / max(mem_ini, 1e-9) * 100.0
    print(f"   - Memoria inicial: {mem_ini:.1f} MB")
    print(f"   - Memoria optimizada: {mem_fin:.1f} MB")
    print(f"   - Reducción de memoria: {red_pct:.1f}%")

    # 3) Split temporal 80/20
    print("   Realizando división temporal del dataset...")
    if "timestamp" in df_opt.columns:
        df_opt["timestamp"] = pd.to_datetime(df_opt["timestamp"], utc=True, errors="coerce")
        df_opt = df_opt.sort_values("timestamp").set_index("timestamp")
    elif not isinstance(df_opt.index, pd.DatetimeIndex):
        df_opt.index = pd.to_datetime(df_opt.index, utc=True, errors="coerce")
        df_opt = df_opt.sort_index()

    n = len(df_opt)
    idx_split = int(n * 0.8)
    fecha_div = df_opt.index[idx_split] if n else pd.NaT
    df_train = df_opt.iloc[:idx_split].copy()
    df_valid = df_opt.iloc[idx_split:].copy()

    print(f"   1. Dataset de entrenamiento: {len(df_train):,} registros ({len(df_train)/n*100:.1f}%)")
    print(f"   2. Dataset de validación: {len(df_valid):,} registros ({len(df_valid)/n*100:.1f}%)")
    print(f"   3. Fecha de división: {fecha_div}")
    if len(df_train):
        print(f"    Entrenamiento: {df_train.index.min()} → {df_train.index.max()}")
    if len(df_valid):
        print(f"    Validación: {df_valid.index.min()} → {df_valid.index.max()}")

    if "falla_30d" in df_opt.columns:
        print(f"   Tasa de fallas en entrenamiento: {df_train['falla_30d'].mean():.3f}")
        print(f"   Tasa de fallas en validación: {df_valid['falla_30d'].mean():.3f}")

    archivos = {}
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # 4) Parquet/CSV
    if save_parquet_csv:
        print("   Guardando datasets en Parquet/CSV...")
        try:
            ruta_features.mkdir(parents=True, exist_ok=True)
            p_all = ruta_features / f"transformer_features_complete_{ts}.parquet"
            c_all = ruta_features / f"transformer_features_complete_{ts}.csv"
            df_opt.to_parquet(p_all, compression="snappy")
            df_opt.to_csv(c_all, index=True)
            archivos["dataset_completo"] = {"parquet": p_all, "csv": c_all, "registros": len(df_opt), "features": df_opt.shape[1]}
            p_train = ruta_features / f"transformer_features_train_{ts}.parquet"
            df_train.to_parquet(p_train, compression="snappy")
            archivos["dataset_train"] = {"parquet": p_train, "registros": len(df_train), "features": df_train.shape[1]}
            p_valid = ruta_features / f"transformer_features_validation_{ts}.parquet"
            df_valid.to_parquet(p_valid, compression="snappy")
            archivos["dataset_validation"] = {"parquet": p_valid, "registros": len(df_valid), "features": df_valid.shape[1]}
            print(f"   - Dataset completo guardado: {df_opt.shape}")
            print(f"   - Dataset entrenamiento guardado: {df_train.shape}")
            print(f"   - Dataset validación guardado: {df_valid.shape}")
        except Exception as e:
            print(f"  Error guardando Parquet/CSV: {e}")
            archivos["error_parquet_csv"] = str(e)

    # 5) Delta Lake (Gold)
    if save_delta:
        write_deltalake = _try_import_write_deltalake()
        if write_deltalake is None:
            print("    'deltalake' no disponible; omitiendo guardado Delta.")
        else:
            try:
                def _save_delta(df_in: pd.DataFrame, path_dir: Path):
                    if df_in.empty: return
                    out = df_in.reset_index().copy()
                    out["year"]  = out["timestamp"].dt.year.astype("int32")
                    out["month"] = out["timestamp"].dt.month.astype("int8")
                    path_dir.mkdir(parents=True, exist_ok=True)
                    write_deltalake(str(path_dir), out, mode="overwrite", partition_by=["year","month"])

                _save_delta(df_opt,   RUTA_GOLD_COMPLETE)
                _save_delta(df_train, RUTA_GOLD_TRAIN)
                _save_delta(df_valid, RUTA_GOLD_VALID)
                archivos["delta"] = {"complete": RUTA_GOLD_COMPLETE, "train": RUTA_GOLD_TRAIN, "valid": RUTA_GOLD_VALID}
                print("    Tablas Delta (Gold) guardadas (particionado year/month)")
            except Exception as e:
                print(f"    Error guardando Delta Gold: {e}")
                archivos["error_delta"] = str(e)

    # 6) Metadatos TXT
    print(" Generando documentación técnica completa...")
    try:
        meta = ruta_features / f"transformer_features_metadata_{ts}.txt"
        with open(meta, "w", encoding="utf-8") as f:
            w = f.write
            w("DOCUMENTACIÓN TÉCNICA - FEATURES DE TRANSFORMADORES ELÉCTRICOS\n")
            w("=" * 70 + "\n\n")
            w(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            w("Notebook/Base: 03_feature_engineering_transformadores\n")
            w("Objetivo: Mantenimiento Predictivo de Transformadores\n")
            w(f"Horizonte de predicción: {parametros['horizonte_prediccion_dias']} días\n\n")
            w("DIMENSIONES DEL DATASET\n" + "-"*25 + "\n")
            w(f"Total de registros: {df_opt.shape[0]:,}\n")
            w(f"Total de features: {df_opt.shape[1]}\n")
            w(f"Período temporal: {df_opt.index.min()} → {df_opt.index.max()}\n")
            w(f"Frecuencia: {parametros['frecuencia_muestreo']} (horaria)\n")
            w(f"Memoria utilizada: {mem_fin:.1f} MB\n\n")
            w("FEATURES POR CATEGORÍA TÉCNICA\n" + "-"*32 + "\n")
            for k, v in cat.items():
                if len(v) > 0:
                    w(f"{k} ({len(v)} features):\n")
                    for i, feat in enumerate(v, 1):
                        w(f"  {i:2d}. {feat}\n")
            w("\nETIQUETAS DE PREDICCIÓN\n" + "-"*22 + "\n")
            w("Etiqueta binaria: falla_30d\nEtiqueta multi-clase: estado_futuro\nRUL: rul_dias\nSeveridad: severidad_futura\n\n")
            w("DIVISIÓN TEMPORAL\n" + "-"*16 + "\n")
            w(f"Entrenamiento: {len(df_train):,} registros ({len(df_train)/n*100:.1f}%)\n")
            w(f"Validación: {len(df_valid):,} registros ({len(df_valid)/n*100:.1f}%)\n")
            w(f"Fecha de división: {fecha_div}\n\n")
            w("PARÁMETROS TÉCNICOS\n" + "-"*18 + "\n")
            for k, v in parametros.items():
                w(f"{k}: {v}\n")
        archivos["metadatos"] = meta
        print("    Documentación técnica generada")
    except Exception as e:
        print(f"    Error generando documentación: {e}")

    # 7) Resumen final
    print("   * FEATURE ENGINEERING COMPLETADO EXITOSAMENTE")
    print("=" * 50)
    print(f"   - Dataset final: {df_opt.shape[0]:,} registros × {df_opt.shape[1]} features")
    print(f"   - Features creados: {len(features_termicos + features_electricos)} especializados")
    print(f"   - Etiquetas de predicción: {len(etiquetas_creadas)} tipos")
    num_ds = len([k for k, v in archivos.items() if isinstance(v, dict) and ("parquet" in v or "complete" in v)])
    print(f"   - Archivos guardados: {num_ds} datasets")
    print("    DATASET LISTO PARA ENTRENAMIENTO DE MODELOS DE ML")
    if "falla_30d" in df_opt.columns:
        p = float(df_opt["falla_30d"].mean())
        print("    ESTADÍSTICAS CLAVE PARA MODELADO:")
        print(f"   - Tasa de casos positivos: {p:.3f} ({p*100:.1f}%)")
        if "rul_dias" in df_opt.columns:
            print(f"   - RUL medio: {df_opt['rul_dias'].mean():.1f} días "
                  f"(rango: {df_opt['rul_dias'].min():.1f} - {df_opt['rul_dias'].max():.1f})")
        if "severidad_futura" in df_opt.columns:
            print(f"   - Severidad media: {df_opt['severidad_futura'].mean():.1f}%")

    return {
        "dataset_final": df_opt,
        "dataset_train": df_train,
        "dataset_validation": df_valid,
        "archivos_guardados": archivos,
        "estadisticas": {
            "total_features": total_features,
            "memoria_mb": mem_fin,
            "reduccion_memoria_pct": red_pct,
            "fecha_division": fecha_div,
        },
    }
