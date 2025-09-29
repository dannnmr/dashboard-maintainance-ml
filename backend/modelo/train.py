# Orchestrates the full training run and saves artifacts/meta

# --- Headless Matplotlib (avoid tkinter warnings) ---
import matplotlib
matplotlib.use("Agg")

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score

from config import (
    RANDOM_STATE, LOOKBACK, HORIZON_SHIFT,
    OPERATE_WITH_AE_ONLY, ALPHA,
    BETA_F, PRECISION_TARGET, SMOOTH_K, SMOOTH_M,
    TARGET
)
from paths import ARTIFACTS_DIR, PLOTS_DIR, RESULTS_DIR
from data_load import load_gold_complete
from prep import (
    select_columns, temporal_split, fit_impute_train_medians,
    apply_impute, build_label_encoder, scale_fit_transform_normal,
    scale_transform, make_sequences
)
from ae import train_ae, recon_error
from iforest import fit_iforest, iforest_scores
from ensemble import (
    minmax_transform, best_thr_fbeta, threshold_for_min_precision,
    smooth_alerts, ensemble_scores, metrics_auc
)
from explain import ae_feature_contribs, surrogate_tree

def save_show(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def run_training():
    # 1) Load Gold
    df = load_gold_complete()

    # 2) Column selection & full matrices
    X_cols = select_columns(df)
    # IMPORTANT: align with notebook — replace inf by NaN before imputing
    X_full = df[X_cols].replace([np.inf, -np.inf], np.nan)
    y_full = df[TARGET].astype(str)

    # 3) Split (80/20) + impute (train medians only)
    X_tr, X_va, y_tr, y_va = temporal_split(X_full, y_full, split_ratio=0.80)
    medians = fit_impute_train_medians(X_tr)
    X_tr = apply_impute(X_tr, medians)
    X_va = apply_impute(X_va, medians)

    # 4) Labels
    le = build_label_encoder(y_full)
    y_tr_enc = le.transform(y_tr)
    y_va_enc = le.transform(y_va)
    try:
        normal_id = int(np.where(le.classes_ == "NORMAL")[0][0])
    except Exception:
        from collections import Counter
        normal_id = Counter(y_tr_enc).most_common(1)[0][0]
    y_valid_bin = (y_va_enc != normal_id).astype(int)

    # 5) AE scaling (fit on NORMAL only)
    scaler_ae, X_tr_sc, mask_normal = scale_fit_transform_normal(X_tr, y_tr_enc, normal_id)
    X_va_sc = scale_transform(scaler_ae, X_va)

    # 6) Sequences
    Xtr_seq, tr_idx = make_sequences(X_tr_sc, LOOKBACK, HORIZON_SHIFT)
    Xva_seq, va_idx = make_sequences(X_va_sc, LOOKBACK, HORIZON_SHIFT)
    yva_bin_aligned = y_valid_bin[LOOKBACK-1:len(y_valid_bin)-HORIZON_SHIFT]

    # 7) AE
    ae_model, hist = train_ae(Xtr_seq, Xva_seq, verbose=1)
    plt.figure(); plt.plot(hist["loss"], label="Train"); plt.plot(hist["val_loss"], label="Valid")
    plt.title("AE-LSTM Training vs Validation Loss"); plt.legend()
    save_show(PLOTS_DIR / "ae_training_loss.png")

    err_tr = recon_error(ae_model, Xtr_seq)
    err_va = recon_error(ae_model, Xva_seq)
    ae_roc, ae_pr = roc_auc_score(yva_bin_aligned, err_va), average_precision_score(yva_bin_aligned, err_va)
    thr_ae_p95 = float(np.quantile(err_tr, 0.95))
    ae_f2, thr_ae_f2, _ = best_thr_fbeta(err_va, yva_bin_aligned, beta=2.0)

    # 8) Isolation Forest
    if_model, if_scaler = fit_iforest(X_tr[mask_normal], random_state=RANDOM_STATE)
    scores_if = iforest_scores(if_model, if_scaler, X_va)           # validation
    scores_if_tr = iforest_scores(if_model, if_scaler, X_tr[mask_normal])  # train-normal

    if_roc, if_pr = roc_auc_score(y_valid_bin, scores_if), average_precision_score(y_valid_bin, scores_if)
    thr_if_p95 = float(np.quantile(scores_if_tr, 0.95))
    if_f2, thr_if_f2, _ = best_thr_fbeta(scores_if, y_valid_bin, beta=2.0)

    # 9) Normalization, ensemble & operating policy
    ae_lo, ae_hi = err_tr.min(), err_tr.max()
    ae_norm = minmax_transform(err_va, ae_lo, ae_hi)

    idx_start = LOOKBACK - 1
    idx_end   = len(scores_if) - HORIZON_SHIFT
    scores_if_aligned = scores_if[idx_start:idx_end]
    assert len(scores_if_aligned) == len(ae_norm) == len(yva_bin_aligned)

    if_lo, if_hi = scores_if_tr.min(), scores_if_tr.max()
    if_norm = minmax_transform(scores_if_aligned, if_lo, if_hi)

    ens_score = ensemble_scores(ae_norm, if_norm, ALPHA)
    ens_roc, ens_pr = metrics_auc(yva_bin_aligned, ens_score)
    ens_f2, thr_ens_f2, _ = best_thr_fbeta(ens_score, yva_bin_aligned, beta=2.0)

    # --- Operating policy (normalized scores) ---
    operate_score = ae_norm if OPERATE_WITH_AE_ONLY else ens_score
    # Threshold = max(F-beta, min-precision)
    _, thr_fbeta, _ = best_thr_fbeta(operate_score, yva_bin_aligned, beta=BETA_F)
    thr_prec = threshold_for_min_precision(operate_score, yva_bin_aligned, min_precision=PRECISION_TARGET)
    operate_thr = max(thr_fbeta, thr_prec)

    yhat_operate = (operate_score > operate_thr).astype(int)
    alert_operate = smooth_alerts(yhat_operate, k=SMOOTH_K, m=SMOOTH_M)

    print("\n[Operative decision]")
    print(classification_report(yva_bin_aligned, alert_operate, target_names=["NORMAL","NO-NORMAL"], digits=4, zero_division=0))
    cm = confusion_matrix(yva_bin_aligned, alert_operate, labels=[0,1])
    ConfusionMatrixDisplay(cm, display_labels=["NORMAL","NO-NORMAL"]).plot(cmap="PuBu")
    plt.title("Operative decision + Smoothing")
    save_show(PLOTS_DIR / "operative_smoothing_cm.png")

    # 10) Explainability (optional plots)
    contrib = ae_feature_contribs(ae_model, Xva_seq, X_cols)
    # Surrogate tree on aligned validation rows
    X_valid_aligned = X_va.iloc[LOOKBACK-1:LOOKBACK-1+len(alert_operate)]
    sur_clf, sur_imp = surrogate_tree(X_valid_aligned, alert_operate)
    plt.figure(); sur_imp.head(12)[::-1].plot(kind="barh")
    plt.title("Surrogate – Importances")
    save_show(PLOTS_DIR / "surrogate_importances.png")

    # 11) Save artifacts
    ae_model.save(ARTIFACTS_DIR / "ae_lstm.keras")
    joblib.dump(if_model, ARTIFACTS_DIR / "iforest.pkl")
    pd.Series(X_cols).to_csv(ARTIFACTS_DIR / "feature_columns.csv", index=False)
    joblib.dump(scaler_ae, ARTIFACTS_DIR / "scaler_ae.pkl")
    joblib.dump(if_scaler, ARTIFACTS_DIR / "scaler_if.pkl")
    joblib.dump(le, ARTIFACTS_DIR / "label_encoder.pkl")

    eval_df = pd.DataFrame({
        "timestamp": va_idx,
        "y_true": yva_bin_aligned,
        "operate_score": operate_score,
        "operate_pred_raw": yhat_operate,
        "operate_pred_smooth": alert_operate,
    }).set_index("timestamp")
    eval_df.to_parquet(ARTIFACTS_DIR / "eval_valid_window.parquet")

    meta = {
        "lookback": LOOKBACK,
        "horizon_shift": HORIZON_SHIFT,
        "classes": le.classes_.tolist(),
        "normal_id": int(normal_id),

        "ae_thr_p95": float(thr_ae_p95),
        "ae_thr_f2": float(thr_ae_f2),
        "ae_roc_auc": float(ae_roc), "ae_pr_auc": float(ae_pr),

        "if_thr_p95": float(thr_if_p95),
        "if_thr_f2": float(thr_if_f2),
        "if_roc_auc": float(if_roc), "if_pr_auc": float(if_pr),

        "alpha": float(ALPHA),
        "ens_thr_f2": float(thr_ens_f2),
        "ens_roc_auc": float(ens_roc), "ens_pr_auc": float(ens_pr),

        "ae_score_min": float(ae_lo), "ae_score_max": float(ae_hi),
        "if_score_min": float(if_lo), "if_score_max": float(if_hi),

        # Operating policy actually used
        "operate_with_ae_only": bool(OPERATE_WITH_AE_ONLY),
        "operate_thr": float(operate_thr),           # normalized [0,1]
        "operate_f_beta": float(BETA_F),
        "operate_precision_target": float(PRECISION_TARGET),
        "smoothing_k": int(SMOOTH_K), "smoothing_m": int(SMOOTH_M)
    }
    with open(ARTIFACTS_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Horizon-tagged copy for comparisons
    with open(RESULTS_DIR / f"meta_h{HORIZON_SHIFT}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n✅ Artifacts saved to:", ARTIFACTS_DIR)
    joblib.dump(medians, ARTIFACTS_DIR / "medians.pkl")
    return meta

if __name__ == "__main__":
    run_training()
