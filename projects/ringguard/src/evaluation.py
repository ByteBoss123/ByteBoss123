"""
RingGuard evaluation utilities.

Reports metrics in the exact vocabulary a fraud/risk team uses day to day:
Precision/Recall, AUC, the Kolmogorov-Smirnov (KS) statistic that separates
good vs. bad score distributions, and false-positive rate at a chosen
operating threshold (since in production you pick a threshold off the
business's tolerance for false positives, not 0.5 by default).
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, roc_curve,
)


def ks_statistic(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def fpr_at_threshold(y_true, y_score, threshold):
    pred = (y_score >= threshold).astype(int)
    fp = ((pred == 1) & (y_true == 0)).sum()
    tn = ((pred == 0) & (y_true == 0)).sum()
    return float(fp / max(1, fp + tn))


def best_threshold_for_target_fpr(y_true, y_score, target_fpr=0.05):
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    # roc_curve prepends a sentinel threshold of max(score)+1 (often inf for
    # tree-model probabilities) purely to anchor the curve at (0,0). It is
    # not a usable operating point -- on small samples it can be the
    # closest match to the target FPR, which silently produces a threshold
    # nothing can ever clear. Drop it before searching.
    if len(thresh) > 1:
        fpr, tpr, thresh = fpr[1:], tpr[1:], thresh[1:]
    dist = np.abs(fpr - target_fpr)
    # Multiple thresholds can tie on distance to the target FPR (e.g. several
    # points all sit at fpr=0 with different recall). Among ties, prefer the
    # highest recall -- that's the efficient-frontier point -- rather than
    # argmin's default of "first index found", which would silently return
    # the worst-recall tie.
    best_dist = dist.min()
    candidates = np.where(dist == best_dist)[0]
    idx = candidates[np.argmax(tpr[candidates])]
    return float(thresh[idx]), float(fpr[idx]), float(tpr[idx])


def evaluate(y_true, y_score, threshold=None, target_fpr=0.05, label=""):
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    ks = ks_statistic(y_true, y_score) if len(set(y_true)) > 1 else float("nan")

    if threshold is None:
        threshold, achieved_fpr, achieved_tpr = best_threshold_for_target_fpr(
            y_true, y_score, target_fpr
        )
    pred = (y_score >= threshold).astype(int)
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    fpr = fpr_at_threshold(y_true, y_score, threshold)

    return {
        "label": label,
        "auc": round(auc, 4),
        "ks_statistic": round(ks, 4),
        "operating_threshold": round(float(threshold), 4),
        "precision_at_threshold": round(float(precision), 4),
        "recall_at_threshold": round(float(recall), 4),
        "fpr_at_threshold": round(float(fpr), 4),
        "n": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
    }
