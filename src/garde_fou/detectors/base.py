"""The contract every detector honours."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np


class Detector(ABC):
    """A scorer, not a classifier.

    `score` returns a value in [0, 1] that increases with the likelihood that
    the text is an injection. Turning that into a block/allow decision is the
    *operator's* job and happens at a threshold calibrated on validation data —
    see `garde_fou.evaluation.metrics.threshold_at_fpr`. Keeping the threshold
    outside the model is what makes "95% accurate" an answerable question
    instead of a marketing claim.
    """

    name: str = "detector"
    requires_fit: bool = True
    # Rough description used in the latency/accuracy table.
    kind: str = "unknown"

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> Detector:
        return self

    @abstractmethod
    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Scores for `texts`, same length and order. `[]` returns `[]`.

        The empty case is part of the contract, not an edge case to guard at
        every call site: a cascade stage can legitimately be handed nothing
        left to decide, and a batch endpoint can be handed an empty page.
        """

    def score_one(self, text: str) -> float:
        return float(self.score([text])[0])

    def explain(self, text: str) -> dict:
        """Optional human-readable reason. Rule engines override this."""
        return {"score": self.score_one(text), "reasons": []}

    # --- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(json.dumps({"name": self.name, "kind": self.kind}))
        self._save(path)

    def _save(self, path: Path) -> None:  # pragma: no cover - overridden
        raise NotImplementedError(f"{type(self).__name__} is not persistable")

    @classmethod
    def load(cls, path: Path) -> Detector:  # pragma: no cover - overridden
        raise NotImplementedError
