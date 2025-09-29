from pathlib import Path

def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for cand in (p, *p.parents):
        if (cand / "data").exists():
            return cand
    return p

BASE_DIR = find_repo_root()

# Data
RUTA_GOLD_COMPLETE = BASE_DIR / "data" / "capa_gold" / "features_transformador" / "features_complete"

# Artifacts
ARTIFACTS_DIR = BASE_DIR / "backend" / "modelo" / "artifacts_anomalia"
PLOTS_DIR     = ARTIFACTS_DIR / "plots"
RESULTS_DIR   = BASE_DIR / "backend" / "modelo" / "artifacts_anomalia_results"
print("Repo root:", BASE_DIR)
print("Gold data path:", RUTA_GOLD_COMPLETE)
print("Artifacts path:", ARTIFACTS_DIR)
print("  - Plots path:", PLOTS_DIR)
print("  - Results path:", RESULTS_DIR)

for d in (ARTIFACTS_DIR, PLOTS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
