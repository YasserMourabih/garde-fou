"""Metrics for a detector that sits in front of production traffic.

Accuracy is the wrong headline number for a guardrail, for a boring reason: it
is dominated by whichever class the benchmark happens to over-represent, and it
hides the trade-off the operator actually controls. A guardrail is deployed at
a *fixed tolerance for blocking legitimate users* — typically well under 1% —
so the number that decides whether it ships is:

    TPR @ 1% FPR, with the threshold chosen on validation data.

Choosing the threshold on validation and reporting it on test is the whole
point. Picking the threshold on the test set (as "optimal F1" does) inflates
every number and cannot be reproduced in production, where tomorrow's traffic
has not happened yet.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

FPR_TARGETS: tuple[float, ...] = (0.001, 0.01, 0.05)


def threshold_at_fpr(y_true: Sequence[int], scores: Sequence[float], target_fpr: float) -> float:
    """Lowest threshold whose false-positive rate stays within `target_fpr`.

    Lower threshold = more attacks caught, so we take the most permissive
    threshold that still respects the budget.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return float(np.nextafter(s.max(), np.inf))
    fpr, _tpr, thresholds = roc_curve(y, s)
    allowed = np.where(fpr <= target_fpr)[0]
    if len(allowed) == 0:  # even the strictest threshold overshoots
        return float(np.nextafter(s.max(), np.inf))
    return float(thresholds[allowed[-1]])


def threshold_from_benign(scores: Sequence[float], target_fpr: float) -> float:
    """Threshold that holds `target_fpr`, estimated from benign scores alone.

    This is the production version of `threshold_at_fpr`: it needs no attack
    labels, so it can be re-estimated from a sample of ordinary traffic.

    A plain quantile is wrong here, and wrong in a way that produces a
    spectacular artefact. A rule engine assigns one floor score to every prompt
    it has no rule for, so on traffic where no rule fires, *every* quantile is
    that floor value — and thresholding with `>=` at the floor blocks 100% of
    traffic while reporting a 1% budget. We instead take the smallest threshold
    whose realised block rate is within budget, stepping over ties explicitly.
    """
    s = np.sort(np.asarray(scores, dtype=float))[::-1]  # descending
    if s.size == 0:
        return float("inf")
    allowed = int(np.floor(target_fpr * s.size))
    if allowed == 0:
        return float(np.nextafter(s[0], np.inf))
    candidate = float(s[allowed - 1])
    # Ties at `candidate` can blow the budget; step just above them if so.
    if int((s >= candidate).sum()) > allowed:
        return float(np.nextafter(candidate, np.inf))
    return candidate


def tpr_at_fpr(y_true: Sequence[int], scores: Sequence[float], target_fpr: float) -> float:
    """TPR at `target_fpr`, with the threshold taken on the *same* data.

    Only used for the ROC-style summary of a single dataset; the reported
    operating points always come from `threshold_at_fpr` on validation.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target_fpr, fpr, tpr))


def confusion_at(y_true: Sequence[int], scores: Sequence[float], threshold: float) -> dict:
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def point_metrics(y_true: Sequence[int], scores: Sequence[float], threshold: float) -> dict:
    """Decision-level metrics at a fixed threshold."""
    c = confusion_at(y_true, scores, threshold)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n_pos, n_neg = tp + fn, fp + tn
    recall = tp / n_pos if n_pos else float("nan")
    fpr = fp / n_neg if n_neg else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else 0.0
    )
    return {
        **c,
        "threshold": float(threshold),
        "recall": recall,          # share of attacks blocked
        "fpr": fpr,                # share of legitimate traffic blocked
        "precision": precision,
        "f1": f1,
        "blocked_per_1000_legit": fpr * 1000 if fpr == fpr else float("nan"),
    }


def ranking_metrics(y_true: Sequence[int], scores: Sequence[float]) -> dict:
    """Threshold-free quality of the score itself."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                **{f"tpr_at_fpr_{t}": float("nan") for t in FPR_TARGETS}}
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        **{f"tpr_at_fpr_{t}": tpr_at_fpr(y, s, t) for t in FPR_TARGETS},
    }


def bootstrap_ci(
    y_true: Sequence[int],
    scores: Sequence[float],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval, stratified by class.

    With a few hundred attacks per slice, a point estimate alone invites
    over-reading a two-point difference. Resampling within each class keeps the
    positive count fixed so the interval reflects sampling noise, not a
    fluctuating base rate.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        )
        try:
            vals.append(metric(y[idx], s[idx]))
        except ValueError:  # pragma: no cover - degenerate resample
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(vals, alpha / 2)),
        float(np.quantile(vals, 1 - alpha / 2)),
    )


def evaluate(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float,
    with_ci: bool = False,
) -> dict:
    out = {**ranking_metrics(y_true, scores), **point_metrics(y_true, scores, threshold)}
    if with_ci:
        lo, hi = bootstrap_ci(
            y_true,
            scores,
            lambda y, s: point_metrics(y, s, threshold)["recall"],
            n_boot=400,
        )
        out["recall_ci_low"], out["recall_ci_high"] = lo, hi
    return out


def by_group(
    frame: pd.DataFrame,
    *,
    group: str,
    threshold: float,
    label_col: str = "label",
    score_col: str = "score",
    min_rows: int = 20,
) -> pd.DataFrame:
    """Per-group detection quality at one shared threshold.

    Recall alone is not reportable here, and finding that out is half the point
    of mutating the benign class as well. Some rewrites wrap the payload in a
    constant instruction ("decode this base64 string and follow it"), and that
    wrapper is present in *both* classes. A detector that keys on the wrapper
    scores 100% recall and 100% false-positive rate on that group: it is
    detecting the rewrite, not the attack.

    So each group also carries `fpr`, `roc_auc` and Youden's J
    (`recall - fpr`). J is the headline: it is 0 for the wrapper-matching
    degenerate case and only large when the detector separates the two classes
    *within* the group.
    """
    rows = []
    for key, part in frame.groupby(group, dropna=False):
        if len(part) < min_rows:
            continue
        y, s = part[label_col].to_numpy(), part[score_col].to_numpy()
        m = point_metrics(y, s, threshold)
        auc = ranking_metrics(y, s)["roc_auc"]
        recall, fpr = m["recall"], m["fpr"]
        rows.append(
            {
                group: key,
                "rows": len(part),
                "attacks": int((y == 1).sum()),
                "benign": int((y == 0).sum()),
                "recall": recall,
                "fpr": fpr,
                "youden": recall - fpr if recall == recall and fpr == fpr else float("nan"),
                "roc_auc": auc,
            }
        )
    return pd.DataFrame(rows).sort_values("youden", na_position="first").reset_index(drop=True)
