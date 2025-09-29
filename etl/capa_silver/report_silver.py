import pandas as pd

def generar_reporte_calidad_transformador(df_final: pd.DataFrame, missing_original: int, interpolated: int, estadisticas_estados: dict):
    missing_final = int(df_final.isna().sum().sum())
    total_valores = int(df_final.size)


    categorias = {
        'Térmica': ['temp', 'oil', 'hot', 'ambient'],
        'Eléctrica': ['current', 'voltage', 'power', 'apparent'],
        'Mecánica': ['position', 'tap'],
    }


    variables_por_categoria = {
        cat: len([c for c in df_final.columns if any(t in c.lower() for t in terms)])
        for cat, terms in categorias.items()
    }
    completitud_total = float(((total_valores - missing_final) / total_valores * 100) if total_valores else 0.0)


    return {
        'completitud_total': completitud_total,
        'estados_operacionales': estadisticas_estados,
        'variables_por_categoria': variables_por_categoria,
        'missing_original': int(missing_original),
        'interpolated': int(interpolated),
        'missing_final': int(missing_final),
        'total_valores': int(total_valores),
    }
