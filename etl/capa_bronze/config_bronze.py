# etl/capa_bronze/config.py
# variables que se usan para la extraccion de datos de los transformadores

import os
from pathlib import Path

# Purpose: central configuration for Bronze (tags, time windows, paths)
TAGS = {
    "temperatura_aceite": "TR1.Top oil temperature",
    "temperatura_ambiente": "TR1.Ambient temperature",
    "temperatura_punto_caliente": "TR1.Hot spot temperature",
    "temperatura_burbujeo": "TR1.Bubbling temperature",
    "temperatura_aceite_OLTC": "TR1.Oil temperature OLTC 1",
    "tap_position": "TR1.Tap position",
    "corriente_carga": "TR1.Load current LV Ph 2",
    "voltaje": "TR1.Voltage (phase - ground) HV Ph 2",
    "potencia_aparente": "TR1.Power (apparent power) 1m",
    "margen_burbujeante":"TR1.Bubbling safety margin",
    "buchholz_desconectado":"TR1.Buchholz Relay RB disconnected",
    "cambio_capacitancia":"TR1.Change of capacitance detected",
    "verificar_conexion_sensor_gas":"TR1.Check connection to the gas sensor",
    "verificar_conexion_profibus":"TR1.Check Profibus connection",
    "nivel_aceite_tanque_principal_max":"TR1.Oil level main tank max",
    "nivel_aceite_tanque_principal_min":"TR1.Oil level main tank min",
    "monitores_falla_fuente_alimentacion":"TR1.Power supply failure monitors",
    "falla_fuente_alimentacion_vdc":"TR1.Power supply failure Vdc",
    "cambio_capacitancia_fase_1":"TR1.Change of cap detection HV Ph 1",
    "cambio_capacitancia_fase_2":"TR1.Change of cap detection HV Ph 2",
    "cambio_capacitancia_fase_3":"TR1.Change of cap detection HV Ph 3",
    "cambio_tan_fase_1":"TR1.Change of tan HV Ph 1",
    "cambio_tan_fase_2":"TR1.Change of tan HV Ph 2",
    "cambio_tan_fase_3":"TR1.Change of tan HV Ph 3",
    "factor_carga_fase_2":"TR1.Load factor",
    "max_sobrecarga_fase_2":"TR1.Maximum overload",
    "min_sobrecarga_fase_2":"TR1.Minimum overload",
    "numero_sobrecargas_fase_2":"TR1.Number of overcurrents",
    "numero_sobretensiones_fase_2":"TR1.Number of overvoltages",
    "voltaje_fase_1":"TR1.Voltage (phase - ground) HV Ph 1",
    "voltaje_fase_3":"TR1.Voltage (phase - ground) HV Ph 3",
    "voltaje_fase_1_fase_2":"TR1.Voltage (phase - phase) HV Ph 1 - Ph 2",
    "voltaje_fase_2_fase_3":"TR1.Voltage (phase - phase) HV Ph 2 - Ph 3",
    "voltaje_fase_3_fase_1":"TR1.Voltage (phase - phase) HV Ph 3 - Ph 1",
    "emergencia_enfriamiento_fallido":"TR1.Emergency loading cooler failed",
    "emergencia_enfriamiento_sobre_carga":"TR1.Emergency loading if overloading",
    "ruptura_membrana_relay":"TR1.Membrane rupture relay",
    "flujo_aceite_relay_OLTC":"TR1.Oil flow relay OLTC",
    "valvula_alivio_presion":"TR1.Pressure relief valve",
    "valvula_alivio_presion_OLTC":"TR1.Pressure relief valve OLTC",
    "motor_cortacircuito_OLTC_ON":"TR1.Circuit breaker motor OLTC ON",
    "motor_OLTC_en_operacion":"TR1.OLTC motor in operation",
    "switch_S21_local":"TR1.Switch S21 local",
    "switch_S22_remote":"TR1.Switch S22 remote",
    "system_condition":"TR1.System condition",
    "acetileno":"TR1.Acetylene (C2H2)",
    "dioxido_carbono":"TR1.Carbon Dioxide (CO2)",
    "monoxido_carbono":"TR1.Carbon Monoxide (CO)",
    "etano":"TR1.Ethane (C2H6)",
    "etileno":"TR1.Ethylene (C2H4)",
    "hidrogeno":"TR1.Hydrogen (H2)",
    "metano":"TR1.Methane (CH4)",
    "nitrogeno":"TR1.Nitrogen (N2)",
    "oxigeno":"TR1.Oxygene (O2)",
    "gradient_acetileno":"TR1.Gradient Acetylene (C2H2)",
    "gradient_dioxido_carbono":"TR1.Gradient Carbon Dioxide (CO2)",
    "gradient_monoxido_carbono":"TR1.Gradient Carbon Monoxide (CO)",
    "gradient_etano":"TR1.Gradient Ethane (C2H6)",
    "gradient_etileno":"TR1.Gradient Ethylene (C2H4)",
    "gradient_hidrogeno":"TR1.Gradient Hydrogen (H2)",
    "gradient_metano":"TR1.Gradient Methane (CH4)",
    "gradient_nitrogeno":"TR1.Gradient Nitrogen (N2)",
    "gradient_oxigeno":"TR1.Gradient Oxygene (O2)",
    "gradient_tdcg":"TR1.Gradient TDCG",
    "tdcg":"TR1.TDCG",
    "breakdown_voltage":"TR1.Breakdown voltage",
    "humedad_papel_aislante":"TR1.Moisture of insulation paper",
    "nivel_aceite_OLTC_max":"TR1.Oil level OLTC max",
    "nivel_aceite_OLTC_min":"TR1.Oil level OLTC min",
    "actividad_agua":"TR1.Water activity",
    "contenido_agua_aceite_OLTC_1":"TR1.Water content in oil OLTC 1",
    "contenido_agua_aceite_OLTC_2":"TR1.Water content in oil OLTC 2",
    "perdidas_actuales":"TR1.Actual losses",
    "consumo_vida_util_absoluto":"TR1.Lifetime consumption (absolute)",
    "consumo_vida_util_ultimo_anio":"TR1.Lifetime consumption during last year",
    "tasa_envejecimiento":"TR1.Rate of ageing",
    "tasa_envejecimiento_30d":"TR1.Rate of ageing (30d)",
    "resistencia_termica":"TR1.Thermal resistance",
    "ventilador_1":"TR1.Fan 1",
    "ventilador_1_mcb":"TR1.Fan 1 MCB",
    "ventilador_2":"TR1.Fan 2",
    "ventilador_2_mcb":"TR1.Fan 2 MCB",
    "ventilador_3_mcb":"TR1.Fan 3 MCB",
    "ventilador_4_mcb":"TR1.Fan 4 MCB",
    "ventilador_5_mcb":"TR1.Fan 5 MCB",
    "ventilador_6_mcb":"TR1.Fan 6 MCB",
    "ventilador_7_mcb":"TR1.Fan 7 MCB",
    "ventilador_8_mcb":"TR1.Fan 8 MCB",
    "ventilador_9_mcb":"TR1.Fan 9 MCB",
    "ventilador_10_mcb":"TR1.Fan 10 MCB",
    "ventilador_11_mcb":"TR1.Fan 11 MCB",
    "ventilador_12_mcb":"TR1.Fan 12 MCB",
    "ventilador_13_mcb":"TR1.Fan 13 MCB",
    "ventilador_14_mcb":"TR1.Fan 14 MCB",
}


# Historical window (can be overridden by incremental resume)
START_TIME = "2024-09-10T00:00:00Z"
END_TIME = "2025-08-28T00:00:00Z" # or "now" if implement streaming later


REPO_ROOT   = Path(__file__).resolve().parents[2]
DATA_DIR    = (REPO_ROOT / "data").resolve()
LOG_DIR = (REPO_ROOT / "logs").resolve()
BRONZE_TABLE = DATA_DIR / "capa_bronze_v2" / "readings_v1"

# Ensure directories exist (safe to call multiple times).
DATA_DIR.mkdir(parents=True, exist_ok=True)
BRONZE_TABLE.parent.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
