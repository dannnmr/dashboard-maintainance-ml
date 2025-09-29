import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve
)

def minmax_transform(x: np.ndarray, lo: float, hi: float):
    eps = 1e-12
    return (x - lo) / max(hi - lo, eps)

def best_thr_fbeta(scores: np.ndarray, y_true: np.ndarray, beta: float = 2.0, grid: int = 400):
    lo, hi = scores.min(), scores.max()
    ths = np.linspace(lo, hi, grid)
    best = (0.0, None, (0,0,0))
    b2 = beta * beta
    for th in ths:
        yhat = (scores > th).astype(int)
        tp = ((y_true==1) & (yhat==1)).sum()
        fp = ((y_true==0) & (yhat==1)).sum()
        fn = ((y_true==1) & (yhat==0)).sum()
        prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
        rec  = tp / (tp + fn) if (tp+fn)>0 else 0.0
        denom = b2*prec + rec
        fbeta = ((1+b2)*prec*rec/denom) if denom>0 else 0.0
        if fbeta > best[0]:
            f1 = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
            best = (fbeta, th, (prec, rec, f1))
    return best

def threshold_for_min_precision(scores: np.ndarray, y_true: np.ndarray, min_precision: float) -> float:
    """Small helper: first threshold whose precision >= min_precision (on validation)."""
    prec, rec, ths = precision_recall_curve(y_true, scores)
    # prec/rec have length N+1; ths length N
    idx = np.where(prec[:-1] >= min_precision)[0]
    if len(idx) > 0:
        return float(ths[idx[0]])
    return float(ths[-1])  # if none satisfies, use the highest threshold

def smooth_alerts(binary_seq: np.ndarray, k: int = 3, m: int = 5) -> np.ndarray:
    out = np.zeros_like(binary_seq)
    count = 0
    win = np.zeros(m, dtype=int)
    for i, v in enumerate(binary_seq):
        if i >= m: count -= win[i % m]
        win[i % m] = v
        count += v
        out[i] = 1 if count >= k else 0
    return out

def ensemble_scores(ae_norm: np.ndarray, if_norm: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * ae_norm + (1.0 - alpha) * if_norm

def metrics_auc(y_true: np.ndarray, scores: np.ndarray):
    return roc_auc_score(y_true, scores), average_precision_score(y_true, scores)
