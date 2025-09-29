# path: silver/rules_silver.py

def definir_criterios_transformador():
    """
    Criterios técnicos para clasificación operacional del transformador.
    Alineado a nombres normalizados EN (coinciden con el pipeline de preprocesamiento):
      - temp_oil_value               (temperatura del aceite)
      - temp_spot_hot_value          (temperatura punto caliente)
      - temp_ambient_value           (temperatura ambiente)
      - current_load_value           (corriente de carga)
      - factor_carga                 (p.u., si existe en el DataFrame)
      - tap_position_value           (posición del OLTC)
    NOTA: Los umbrales (normal/alerta/crítico) se mantienen iguales a tu versión.
    """

    # --- Criterios individuales (umbrales) ---
    criterios = {
        # TÉRMICOS (°C)
        'temp_oil_value': {
            'normal':  {'min': 35.8,  'max': 65.3},
            'alerta':  {'min': 65.3,  'max': 75.0},
            'critico': {'min': 75.0,  'max': float('inf')},
            'descripcion': 'Temperatura del aceite dieléctrico'
        },
        'temp_spot_hot_value': {
            'normal':  {'min': 38.7,  'max': 76.2},
            'alerta':  {'min': 76.2,  'max': 100.0},
            'critico': {'min': 100.0, 'max': float('inf')},
            'descripcion': 'Temperatura del punto caliente (hot-spot)'
        },
        'temp_ambient_value': {
            'normal':  {'min': -10.0, 'max': 40.0},
            'alerta':  {'min': 40.0,  'max': 50.0},
            'critico': {'min': 50.0,  'max': float('inf')},
            'descripcion': 'Temperatura ambiente (criterio estacional)'
        },

        # ELÉCTRICAS
        'current_load_value': {
            'normal':  {'min': 0.0,  'max': 1508.1},
            'alerta':  {'min': 1508.1, 'max': 1809.7},
            'critico': {'min': 1809.7, 'max': float('inf')},
            'descripcion': 'Corriente de carga (A) en lado BT'
        },
        'factor_carga': {
            'normal':  {'min': 0.0,  'max': 1.0},
            'alerta':  {'min': 1.0,  'max': 1.2},
            'critico': {'min': 1.2,  'max': float('inf')},
            'descripcion': 'Factor de carga (p.u.)'
        },

        # MECÁNICAS
        'tap_position_value': {
            'normal':  {'min': 6.0,  'max': 12.0},
            'alerta':  {'min': 3.0,  'max': 15.0},
            'critico': {'min': 0.0,  'max': 17.0},
            'descripcion': 'Posición del tap OLTC (0–17)'
        }
    }

    # --- Criterios combinados (misma lógica, variables en EN) ---
    criterios_combinados = {
        'sobrecarga_termica': {
            'variables': ['current_load_value', 'temp_oil_value'],
            'condicion': 'ambas en alerta o crítico',
            'severidad': 'critico',
            'descripcion': 'Sobrecarga con calentamiento excesivo'
        },
        'estres_termico': {
            'variables': ['temp_spot_hot_value', 'temp_oil_value'],
            'condicion': 'diferencia >25°C',
            'severidad': 'alerta',
            'descripcion': 'Gradiente térmico excesivo'
        },
        'regulacion_inestable': {
            'variables': ['tap_position_value'],
            'condicion': 'cambios frecuentes',
            'severidad': 'alerta',
            'descripcion': 'Regulación de voltaje inestable'
        }
    }

    return criterios, criterios_combinados
