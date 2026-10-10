"""Train, score, evaluate — and keep the raw scores.

Scoring is the expensive step, so every score is written once to
`reports/scores.parquet` in long format. Tables, figures and any later
re-analysis read that file instead of re-running the models, which is what
makes the report regenerable in seconds.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from garde_fou.data.build import load_split
from garde_fou.detectors import build, detector_class
from garde_fou.detectors.base import Detector
from garde_fou.evaluation.metrics import FPR_TARGETS, by_group, evaluate, threshold_at_fpr
from garde_fou.paths import ARTIFACTS, REPORTS, ensure_dirs

log = logging.getLogger(__name__)

TEST_SETS = ("test_in", "test_ood", "test_fr", "test_adv")
CALIBRATION_FPR = 0.01
# Order matters: it is the narrative of the report, cheapest defence first.
DEFAULT_DETECTORS = ("heuristic_raw", "heuristic", "tfidf", "embedding", "transformer")


def train_detector(name: str, *, save: bool = True) -> Detector:
    detector = build(name)
    train = load_split("train")
    if detector.requires_fit:
        log.info("fitting %s on %d rows", name, len(train))
        kwargs = {}
        if name == "transformer":
            val = load_split("val")
            kwargs = {
                "val_texts": val["text"].tolist(),
                "val_labels": val["label"].tolist(),
            }
        detector.fit(train["text"].tolist(), train["label"].tolist(), **kwargs)
    if save:
        path = ARTIFACTS / name
        try:
            detector.save(path)
            log.info("saved %s -> %s", name, path)
        except NotImplementedError:  # pragma: no cover
            log.warning("%s is not persistable, keeping in memory", name)
    return detector


def load_or_train(name: str) -> Detector:
    """Load `name` from `artifacts/`, or fit it if there is nothing usable there."""
    path = ARTIFACTS / name
    if path.exists():
        try:
            log.info("loading %s from %s", name, path)
            return detector_class(name).load(path)
        except (NotImplementedError, FileNotFoundError, KeyError) as exc:
            log.info("%s has no usable artefact (%s), rebuilding", name, exc)
    if name == "cascade":
        # The cascade is assembled from already-trained stages, not fitted.
        return build(name)
    return train_detector(name)


def score_splits(detector: Detector, *, splits: tuple[str, ...] = ("val", *TEST_SETS)) -> pd.DataFrame:
    frames = []
    for split in splits:
        df = load_split(split).reset_index(drop=True)
        log.info("scoring %s on %s (%d rows)", detector.name, split, len(df))
        scores = detector.score(df["text"].tolist())
        out = pd.DataFrame(
            {
                "detector": detector.name,
                "eval_set": split,
                "row": np.arange(len(df)),
                "label": df["label"].astype(int),
                "score": scores,
                "source": df["source"],
                "lang": df["lang"],
                "family": df.get("family", pd.Series(["natural"] * len(df))),
                "mutator": df.get("mutator", pd.Series(["none"] * len(df))),
                "fidelity": df.get("fidelity", pd.Series(["high"] * len(df))),
            }
        )
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def collect_scores(detector_names: tuple[str, ...] = DEFAULT_DETECTORS) -> pd.DataFrame:
    ensure_dirs()
    frames = []
    for name in detector_names:
        detector = load_or_train(name)
        frames.append(score_splits(detector))
    scores = pd.concat(frames, ignore_index=True)
    path = REPORTS / "scores.parquet"
    scores.to_parquet(path, index=False)
    log.info("wrote %s (%d rows)", path.name, len(scores))
    return scores


def load_scores() -> pd.DataFrame:
    path = REPORTS / "scores.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `garde-fou eval` first")
    return pd.read_parquet(path)


def summarise(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the headline table, the robustness table and the raw metric dump."""
    summary_rows: list[dict] = []
    metrics: dict = {}
    thresholds: dict[str, dict[str, float]] = {}

    for detector, part in scores.groupby("detector", sort=False):
        val = part[part.eval_set == "val"]
        thr = {
            f"fpr_{t}": threshold_at_fpr(val["label"], val["score"], t) for t in FPR_TARGETS
        }
        thresholds[detector] = thr
        operating = thr[f"fpr_{CALIBRATION_FPR}"]
        metrics[detector] = {"thresholds": thr, "sets": {}}

        for eval_set in TEST_SETS:
            sub = part[part.eval_set == eval_set]
            if sub.empty:
                continue
            res = evaluate(sub["label"], sub["score"], threshold=operating, with_ci=True)
            metrics[detector]["sets"][eval_set] = res
            summary_rows.append(
                {
                    "detector": detector,
                    "eval_set": eval_set,
                    "roc_auc": res["roc_auc"],
                    "pr_auc": res["pr_auc"],
                    "recall": res["recall"],
                    "recall_ci_low": res["recall_ci_low"],
                    "recall_ci_high": res["recall_ci_high"],
                    "fpr": res["fpr"],
                    "blocked_per_1000_legit": res["blocked_per_1000_legit"],
                    "attacks": res["tp"] + res["fn"],
                    "benign": res["fp"] + res["tn"],
                }
            )

    summary = pd.DataFrame(summary_rows)

    robustness_rows = []
    for detector, part in scores.groupby("detector", sort=False):
        adv = part[part.eval_set == "test_adv"]
        if adv.empty:
            continue
        operating = thresholds[detector][f"fpr_{CALIBRATION_FPR}"]
        table = by_group(adv, group="mutator", threshold=operating)
        fid = adv[["mutator", "fidelity", "family"]].drop_duplicates("mutator")
        table = table.merge(fid, on="mutator", how="left")
        table.insert(0, "detector", detector)
        robustness_rows.append(table)
    robustness = (
        pd.concat(robustness_rows, ignore_index=True) if robustness_rows else pd.DataFrame()
    )

    return summary, robustness, {"thresholds": thresholds, "metrics": metrics}


def run_eval(detector_names: tuple[str, ...] = DEFAULT_DETECTORS, *, reuse: bool = False):
    scores = load_scores() if reuse else collect_scores(detector_names)
    summary, robustness, raw = summarise(scores)
    summary.to_csv(REPORTS / "summary.csv", index=False)
    robustness.to_csv(REPORTS / "robustness.csv", index=False)
    (REPORTS / "metrics.json").write_text(json.dumps(raw, indent=2, default=float))
    log.info("wrote summary.csv, robustness.csv, metrics.json")
    return summary, robustness
