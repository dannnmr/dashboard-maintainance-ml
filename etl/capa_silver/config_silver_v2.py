# path: silver/config_silver.py
from pathlib import Path

def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for cand in (p, *p.parents):
        if (cand / "data").exists():
            return cand
    return p



# Paths
BASE_DIR = find_repo_root()
RUTA_BRONZE = BASE_DIR / "data" / "capa_bronze_v2" / "readings_v1"
RUTA_SILVER = BASE_DIR / "data" / "capa_silver" / "preprocesamiento_silver"
RUTA_PROCESSED = BASE_DIR / "data" / "processed"
# RUTA_INTERIM = BASE_DIR / "data" / "interim"
# RUTA_REPORTS = BASE_DIR / "reports"

print("Repo root:", BASE_DIR)
print("Bronze data path:", RUTA_BRONZE)
print("Silver data path:", RUTA_SILVER)
print("Processed data path:", RUTA_PROCESSED)
# print("Interim data path:", RUTA_INTERIM)
# print("Reports path:", RUTA_REPORTS)

for ruta in [RUTA_SILVER, RUTA_PROCESSED]:
    ruta.mkdir(parents=True, exist_ok=True)


# Specs y rangos (ajusta según tu equipo)
SPECS_TRANSFORMADOR = {
"potencia_nominal_mva": 40,
"tension_primaria_kv": 23,
"tension_secundaria_v": 220,
"frecuencia_hz": 50,
"tipo_refrigeracion": "ONAN",
"rango_tap": (1, 17),
"tap_central": 9,
"corriente_nominal_a": 1050,
"temp_aceite_max_c": 65,
"temp_punto_caliente_max_c": 110,
"temp_ambiente_nominal_c": 40,
}


RANGOS_NORMALES = {
"temperatura_aceite": (20, 65),
"temperatura_aceite_OLTC": (20, 60),
"temperatura_ambiente": (10, 45),
"temperatura_burbujeo": (60, 200),
"temperatura_punto_caliente": (30, 110),
"voltaje": (100, 150),
"corriente_carga": (0, 1200),
"potencia_aparente": (0, 50),
"tap_position": (1, 17),
}

# None = sin límite; si pones un entero, limita a ese número de días
CONSOLIDACION_MAX_DIAS = None
