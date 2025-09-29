# path: gold/__main__.py
import argparse
from pipeline_gold import run_gold_pipeline

def main():
    p = argparse.ArgumentParser(description="Capa GOLD — Feature Engineering de Transformadores")
    p.add_argument("--no-parquet", action="store_true", help="No guardar Parquet/CSV")
    p.add_argument("--no-delta",   action="store_true", help="No guardar Delta Lake")
    args = p.parse_args()

    res = run_gold_pipeline(
        save_parquet_csv=not args.no_parquet,
        save_delta=not args.no_delta,
    )
    # Nada más que imprimir: toda la pipeline ya imprime resúmenes.

if __name__ == "__main__":
    main()
