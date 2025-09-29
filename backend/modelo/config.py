# Modeling configuration (single source of truth)

# Seeds / windows
RANDOM_STATE   = 42
LOOKBACK       = 24       # steps per sequence
HORIZON_SHIFT  = 12       # prediction horizon in hours (0/6/12/24...)

# Operating policy (production-facing)
OPERATE_WITH_AE_ONLY = True  # Recommended for H=12
ALPHA          = 0.9         # AE weight if ensemble is used (ignored when AE-only)

# Threshold policy
BETA_F = 1.0                 # use F1 (balance) instead of F2 for operation
PRECISION_TARGET = 0.60      # minimum precision to accept (raise if too many false alarms)

# Smoothing (raise an alert if >= K anomalies in the last M windows)
SMOOTH_K = 4
SMOOTH_M = 7

# Columns to exclude (labels / text / future-derived)
LEAK_OR_TEXT_COLS = [
    "estado_futuro","falla_30d","rul_dias","severidad_futura",
    "dias_proximo_evento","proximidad_evento","riesgo_acumulativo",
    "estado_operacional","nivel_severidad","variables_anomalas","descripcion_anomalia"
]
TARGET = "estado_futuro"
