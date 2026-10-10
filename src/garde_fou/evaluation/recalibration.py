"""Does the operating point survive the move to production traffic?

The audit's most uncomfortable result is not that detection drops out of
distribution — everyone expects some of that — it is that the *threshold* stops
meaning what it said. A threshold picked at a 1% false-positive rate on the
training corpus's validation split turns out to block far more than 1% of real
French traffic, because the score distribution of benign text moved.

That has a cheap fix, and this module measures it: re-estimate the threshold on
a sample of the traffic you actually serve. No retraining, no labels for the
attack class — just benign requests, which any deployment has in abundance.

`pool_fr_adv` is used as the "traffic sample" and `test_fr` as the test set;
they are disjoint slices of the same corpus (see `data/build.py`), so this is a
genuine held-out measurement rather than fitting the threshold on the test set.
"""

from __future__ import annotations

import logging

import pandas as pd

from garde_fou.data.build import load_split
from garde_fou.evaluation.metrics import (
    point_metrics,
    threshold_at_fpr,
    threshold_from_benign,
)
from garde_fou.evaluation.runner import load_or_train
from garde_fou.paths import REPORTS

log = logging.getLogger(__name__)

TARGET_FPR = 0.01


def recalibration_study(
    detector_names: tuple[str, ...],
    *,
    target_fpr: float = TARGET_FPR,
    eval_set: str = "test_fr",
    traffic_split: str = "pool_fr_adv",
) -> pd.DataFrame:
    val = load_split("val")
    traffic = load_split(traffic_split)
    test = load_split(eval_set)

    rows = []
    for name in detector_names:
        detector = load_or_train(name)
        log.info("scoring %s for the recalibration study", name)
        val_scores = detector.score(val["text"].tolist())
        traffic_scores = detector.score(traffic["text"].tolist())
        test_scores = detector.score(test["text"].tolist())
        y_test = test["label"].to_numpy()

        # (a) the threshold a team would pick from its own validation split
        thr_val = threshold_at_fpr(val["label"], val_scores, target_fpr)
        # (b) the same budget, estimated on benign production traffic — no
        #     attack labels needed, which is what makes this deployable.
        thr_traffic = threshold_from_benign(traffic_scores, target_fpr)

        for label, thr in (("validation split", thr_val), (f"{traffic_split} traffic", thr_traffic)):
            m = point_metrics(y_test, test_scores, thr)
            rows.append(
                {
                    "detector": name,
                    "threshold_from": label,
                    "threshold": thr,
                    "recall": m["recall"],
                    "fpr": m["fpr"],
                    "blocked_per_1000_legit": m["blocked_per_1000_legit"],
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "recalibration.csv", index=False)
    log.info("wrote recalibration.csv")
    return table
