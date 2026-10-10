"""Put every detector's score on one interpretable scale.

A rule engine's weighted sum, a logistic regression's probability and a
fine-tuned encoder's softmax are three different numbers. 0.9 means nothing in
common across them, which makes a cascade impossible to threshold and makes
"our model scores 0.87" an unanswerable claim.

So we re-express every score as **the fraction of legitimate validation traffic
it ranks above**. A calibrated score of 0.99 means: "blocking here blocks 1% of
legitimate traffic." One threshold is then meaningful for every stage, and the
threshold *is* the policy — it is stated in the unit the operator cares about.
"""

from __future__ import annotations

import numpy as np


class FprCalibrator:
    """Maps a raw score to `1 - FPR`, estimated on benign validation traffic."""

    def __init__(self) -> None:
        self._benign: np.ndarray | None = None

    def fit(self, scores, labels) -> FprCalibrator:
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels).astype(int)
        benign = scores[labels == 0]
        if benign.size == 0:
            raise ValueError("calibration needs benign examples")
        self._benign = np.sort(benign)
        return self

    @property
    def fitted(self) -> bool:
        return self._benign is not None

    def transform(self, scores) -> np.ndarray:
        if self._benign is None:
            raise RuntimeError("calibrator not fitted")
        s = np.asarray(scores, dtype=float)
        # Share of benign traffic scoring strictly below `s`.
        return np.searchsorted(self._benign, s, side="left") / self._benign.size

    def threshold_for_fpr(self, target_fpr: float) -> float:
        """The calibrated threshold that admits at most `target_fpr`."""
        return 1.0 - target_fpr


class CalibratedDetector:
    """Wrap a single detector so its score means the same thing as a cascade's.

    Without this, a threshold of 0.99 is "1% of legitimate traffic" for the
    cascade and an arbitrary number for everything else — so the API would be
    reporting a guarantee it does not provide.
    """

    kind = "calibrated"

    def __init__(self, detector, calibrator: FprCalibrator | None = None) -> None:
        self.inner = detector
        self.calibrator = calibrator or FprCalibrator()
        self.name = f"{detector.name}+calibrated"
        self.stage_names = [detector.name]

    @property
    def calibrated(self) -> bool:
        return self.calibrator.fitted

    def calibrate(self, texts, labels) -> CalibratedDetector:
        self.calibrator.fit(self.inner.score(texts), labels)
        return self

    def score(self, texts) -> np.ndarray:
        raw = self.inner.score(texts)
        return self.calibrator.transform(raw) if self.calibrated else raw

    def score_one(self, text: str) -> float:
        return float(self.score([text])[0])

    def explain(self, text: str) -> dict:
        detail = dict(self.inner.explain(text))
        detail["score"] = self.score_one(text)
        return detail

    def decision(self, text: str, *, threshold: float = 0.99) -> dict:
        detail = self.explain(text)
        return {
            "blocked": detail["score"] >= threshold,
            "score": detail["score"],
            "calibrated": self.calibrated,
            "legit_traffic_blocked_at_threshold": (
                round(1.0 - threshold, 5) if self.calibrated else None
            ),
            "stages_used": self.stage_names,
            "reasons": detail.get("reasons", []),
        }
