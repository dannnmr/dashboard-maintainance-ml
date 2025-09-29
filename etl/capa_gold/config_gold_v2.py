# path: gold/config_gold.py

PARAMETROS_TRANSFORMADOR: dict = {
    "horizonte_prediccion_dias": 30,
    "ventana_analisis_dias": 90,
    "frecuencia_muestreo": "H",

    # Rolling windows (horas)
    "ventanas_rolling": [24, 72, 168, 720],

    # Lags (horas)
    "lags_corto_plazo":  [1, 6, 12, 24],
    "lags_mediano_plazo": [48, 72, 168],
    "lags_largo_plazo":  [336, 720],

    # Especificaciones nominales
    "capacidad_nominal_mva": 100,
    "voltaje_nominal_kv": 138,
    "temp_aceite_nominal": 65,
    "temp_ambiente_nominal": 40,

    # Umbrales indicadores compuestos
    "umbral_gradiente_termico": 25,
    "umbral_sobrecarga": 1.2,
    "umbral_tap_excesivo": 5,

    # Opciones dominio frecuencia (reservado)
    "freq_nyquist": 0.5,
    "n_componentes_fft": 10,
}
