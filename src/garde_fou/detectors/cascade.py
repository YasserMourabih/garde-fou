"""A staged detector: pay for the expensive model only when it changes the answer.

Guardrail traffic is overwhelmingly benign and overwhelmingly easy, so running
a transformer on "what's the capital of Peru?" buys nothing and costs every user
latency. The cascade escalates only while the current stage is undecided.

**The permission that matters.** A first version of this let every stage end the
cascade in either direction, and it scored exactly like the rule engine alone —
because the rule engine is high-precision and low-recall: it assigns its floor
score to everything it has no rule for, so every attack it did not recognise was
"confidently benign" and never reached a later stage. A stage may only make the
call it is actually good at. The rule engine can **accuse**, never **exonerate**:

| stage | may block early | may allow early |
|---|---|---|
| `heuristic`  | yes — a fired rule is strong evidence | **no** — silence is not evidence |
| `tfidf`      | yes | yes — cheap, and well separated in-domain |
| final stage  | decides everything still pending |

Scores are FPR-calibrated (see `calibration.py`), so one pair of bounds applies
to every stage, and `escalation_` records how often each was reached — the
latency saving is measured, not assumed.

**The final stage has the last word.** Below `block_at`, an earlier stage's
score is discarded in favour of the more capable stage's verdict rather than
combined with it. Taking a maximum across stages would be tempting — the scores
are on a common scale, after all — but it turns the cascade into a union of
detectors and silently multiplies the false-positive rate each stage was
calibrated for. Above `block_at` an early stage *does* decide, which is where
the rule engine contributes its recall.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from garde_fou.detectors.base import Detector
from garde_fou.detectors.calibration import FprCalibrator


@dataclass(frozen=True)
class StagePolicy:
    """Which early exits a stage is trusted to make."""

    name: str
    may_block_early: bool = True
    may_allow_early: bool = True


# Default pipeline. The last stage's permissions are irrelevant: it decides
# everything that reaches it.
#
# `embedding` rather than `transformer` as the final stage is an evidence-based
# choice, not a default: on this project's own numbers the fine-tuned encoder
# wins in-distribution and then blocks ~12% of legitimate traffic on corpora it
# was not trained on, while the frozen multilingual encoder holds its 1% budget
# and transfers to French. See reports/results.md.
DEFAULT_POLICY: tuple[StagePolicy, ...] = (
    StagePolicy("heuristic", may_block_early=True, may_allow_early=False),
    StagePolicy("tfidf", may_block_early=True, may_allow_early=True),
    StagePolicy("embedding"),
)


class CascadeDetector(Detector):
    name = "cascade"
    kind = "cascade"

    def __init__(
        self,
        stages: Sequence[Detector] | None = None,
        *,
        policy: Sequence[StagePolicy] = DEFAULT_POLICY,
        block_at: float = 0.995,
        allow_at: float = 0.90,
    ) -> None:
        self.policy = list(policy)
        self.stage_names = [p.name for p in self.policy]
        self.stages: list[Detector] = list(stages) if stages else []
        self.calibrators: list[FprCalibrator] = []
        self.block_at = block_at
        self.allow_at = allow_at
        self.escalation_: dict[str, int] = {}

    @property
    def requires_fit(self) -> bool:
        return not self.calibrators

    # --- construction -----------------------------------------------------
    @classmethod
    def assemble(
        cls,
        *,
        policy: Sequence[StagePolicy] = DEFAULT_POLICY,
        val_texts: Sequence[str] | None = None,
        val_labels: Sequence[int] | None = None,
        **kwargs,
    ) -> CascadeDetector:
        """Load each stage from `artifacts/` and calibrate it on validation data."""
        from garde_fou.data.build import load_split
        from garde_fou.evaluation.runner import load_or_train

        stages = [load_or_train(p.name) for p in policy]
        obj = cls(stages, policy=policy, **kwargs)
        if val_texts is None or val_labels is None:
            val = load_split("val")
            val_texts, val_labels = val["text"].tolist(), val["label"].tolist()
        obj.calibrate(val_texts, val_labels)
        return obj

    def calibrate(self, texts: Sequence[str], labels: Sequence[int]) -> CascadeDetector:
        self.calibrators = [
            FprCalibrator().fit(stage.score(texts), labels) for stage in self.stages
        ]
        return self

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> CascadeDetector:
        return self.calibrate(texts, labels)

    # --- inference --------------------------------------------------------
    def score(self, texts: Sequence[str]) -> np.ndarray:
        if not self.calibrators:
            raise RuntimeError("cascade must be calibrated before scoring")
        n = len(texts)
        if n == 0:
            return np.empty(0, dtype=float)

        out = np.zeros(n, dtype=float)
        pending = np.arange(n)
        self.escalation_ = {}

        for depth, (stage, calibrator, policy) in enumerate(
            zip(self.stages, self.calibrators, self.policy)
        ):
            if pending.size == 0:
                break
            calibrated = calibrator.transform(stage.score([str(texts[i]) for i in pending]))
            out[pending] = calibrated
            self.escalation_[policy.name] = int(pending.size)

            if depth == len(self.stages) - 1:
                break
            decided = np.zeros(pending.size, dtype=bool)
            if policy.may_block_early:
                decided |= calibrated >= self.block_at
            if policy.may_allow_early:
                decided |= calibrated <= self.allow_at
            pending = pending[~decided]

        return out

    def escalation_rates(self) -> dict[str, float]:
        """Share of the last scored batch that reached each stage."""
        if not self.escalation_:
            return {}
        total = max(self.escalation_.values())
        return {name: count / total for name, count in self.escalation_.items()}

    def explain(self, text: str) -> dict:
        score = self.score_one(text)
        detail = {"score": score, "stages_used": list(self.escalation_), "reasons": []}
        first = self.stages[0]
        if hasattr(first, "explain"):
            detail["reasons"] = first.explain(text).get("reasons", [])
        return detail

    def decision(self, text: str, *, threshold: float = 0.99) -> dict:
        """What the service returns: a verdict plus why, in operator units."""
        detail = self.explain(text)
        return {
            "blocked": detail["score"] >= threshold,
            "score": detail["score"],
            "calibrated": True,
            "legit_traffic_blocked_at_threshold": round(1.0 - threshold, 5),
            "stages_used": detail["stages_used"],
            "reasons": detail["reasons"],
        }

    # --- persistence ------------------------------------------------------
    def _save(self, path: Path) -> None:
        import pickle

        (path / "config.json").write_text(
            json.dumps(
                {
                    "policy": [
                        {
                            "name": p.name,
                            "may_block_early": p.may_block_early,
                            "may_allow_early": p.may_allow_early,
                        }
                        for p in self.policy
                    ],
                    "block_at": self.block_at,
                    "allow_at": self.allow_at,
                }
            )
        )
        with (path / "calibrators.pkl").open("wb") as fh:
            pickle.dump(self.calibrators, fh, protocol=5)

    @classmethod
    def load(cls, path: Path) -> CascadeDetector:
        import pickle

        from garde_fou.evaluation.runner import load_or_train

        cfg = json.loads((path / "config.json").read_text())
        policy = tuple(StagePolicy(**entry) for entry in cfg["policy"])
        obj = cls(
            [load_or_train(p.name) for p in policy],
            policy=policy,
            block_at=cfg["block_at"],
            allow_at=cfg["allow_at"],
        )
        with (path / "calibrators.pkl").open("rb") as fh:
            obj.calibrators = pickle.load(fh)
        return obj
