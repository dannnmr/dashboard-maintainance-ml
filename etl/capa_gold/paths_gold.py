# path: gold/paths_gold.py
from pathlib import Path

def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for cand in (p, *p.parents):
        if (cand / "data").exists():
            return cand
    return p

BASE_DIR = find_repo_root()

# SILVER (entrada) y processed (fallback)
RUTA_SILVER    = BASE_DIR / "data" / "capa_silver" / "preprocesamiento_silver"
RUTA_PROCESSED = BASE_DIR / "data" / "processed"

# GOLD (salida)
RUTA_GOLD_BASE     = BASE_DIR / "data" / "capa_gold" / "features_transformador"
RUTA_GOLD_COMPLETE = RUTA_GOLD_BASE / "features_complete"
RUTA_GOLD_TRAIN    = RUTA_GOLD_BASE / "features_train"
RUTA_GOLD_VALID    = RUTA_GOLD_BASE / "features_valid"
RUTA_GOLD_TEST     = RUTA_GOLD_BASE / "features_test"  # reservado

# Reports (opcional, artefactos legibles humanos)
RUTA_REPORTS_FE = BASE_DIR / "reports" / "feature_engineering"

for ruta in [RUTA_GOLD_BASE, RUTA_GOLD_COMPLETE, RUTA_GOLD_TRAIN, RUTA_GOLD_VALID, RUTA_GOLD_TEST, RUTA_REPORTS_FE]:
    ruta.mkdir(parents=True, exist_ok=True)
