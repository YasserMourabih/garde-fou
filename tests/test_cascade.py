"""Regression tests for the cascade's escalation rules.

The first version of the cascade scored exactly like its rule-engine stage,
because a high-precision/low-recall stage was allowed to end the cascade with
"benign" — and it assigns its floor score to everything it has no rule for. The
tests below pin that behaviour down so it cannot come back.
"""

import numpy as np
import pytest

from garde_fou.detectors.base import Detector
from garde_fou.detectors.cascade import CascadeDetector, StagePolicy


class KeywordStage(Detector):
    """High precision, no recall: fires on one keyword, floors everything else."""

    requires_fit = False
    kind = "stub"

    def __init__(self, name, keyword, hit=0.99, miss=0.01):
        self.name = name
        self.keyword = keyword
        self.hit = hit
        self.miss = miss
        self.calls = 0

    def score(self, texts):
        self.calls += len(texts)
        return np.array(
            [self.hit if self.keyword in str(t).lower() else self.miss for t in texts]
        )


# Calibration traffic: 400 benign prompts, one of which trips the first stage.
# The size matters: a rule that fires on 1 in 400 legitimate prompts calibrates
# to 0.9975, which clears `block_at=0.995`. At 1 in 100 it would calibrate to
# 0.99 and *correctly* fail to clear the bound — you cannot block at a 0.5%
# budget using a rule that costs 1%.
CAL_TEXTS = [f"benign prompt number {i}" for i in range(399)] + ["rule-word here"]
CAL_LABELS = [0] * 400


def _cascade(first_may_allow: bool):
    stages = [
        KeywordStage("heuristic", "rule-word"),
        KeywordStage("tfidf", "subtle-attack"),
    ]
    policy = (
        StagePolicy("heuristic", may_block_early=True, may_allow_early=first_may_allow),
        StagePolicy("tfidf"),
    )
    cascade = CascadeDetector(stages, policy=policy, block_at=0.995, allow_at=0.90)
    cascade.calibrate(CAL_TEXTS, CAL_LABELS)
    for stage in stages:  # calibration is not inference; don't count it
        stage.calls = 0
    return cascade, stages


def test_a_stage_that_cannot_exonerate_lets_the_next_stage_see_the_attack():
    """The regression: an attack only the second stage recognises must be caught."""
    cascade, stages = _cascade(first_may_allow=False)
    score = cascade.score_one("a subtle-attack the rules do not know")
    assert score > 0.9, "second stage never got to see the prompt"
    assert stages[1].calls == 1


def test_allowing_the_rule_engine_to_exonerate_reintroduces_the_bug():
    """Documents why the permission exists, by showing the failure it prevents."""
    cascade, stages = _cascade(first_may_allow=True)
    score = cascade.score_one("a subtle-attack the rules do not know")
    assert score <= 0.9
    assert stages[1].calls == 0, "stage 2 should have been short-circuited here"


def test_an_obvious_attack_stops_at_the_first_stage():
    cascade, stages = _cascade(first_may_allow=False)
    assert cascade.score_one("rule-word appears here") >= 0.995
    assert stages[1].calls == 0, "no need to escalate what the rules already caught"


def test_escalation_rates_are_recorded():
    cascade, _ = _cascade(first_may_allow=False)
    cascade.score(["rule-word here"] * 4 + ["something else"] * 6)
    rates = cascade.escalation_rates()
    assert rates["heuristic"] == pytest.approx(1.0)
    # The four rule hits exit early; the other six escalate.
    assert rates["tfidf"] == pytest.approx(0.6)


def test_the_final_stage_has_the_last_word():
    """Deliberate semantics: a weak early signal is discarded, not combined.

    Below `block_at` the cascade prefers the more capable stage's verdict over
    an earlier stage's hunch. Taking a maximum across stages instead would turn
    the cascade into a union of detectors and quietly multiply the
    false-positive rate each stage was calibrated for.
    """
    cascade, stages = _cascade(first_may_allow=False)
    cascade.block_at = 0.999  # above the rule's calibrated 0.9975, so it cannot block
    score = cascade.score_one("rule-word appears here")
    assert stages[1].calls == 1, "the prompt should have escalated"
    assert score < 0.9, "the final stage's verdict must win below the block bound"


def test_a_rule_too_costly_to_trust_cannot_block_early():
    """A rule that trips 1% of legitimate traffic cannot clear a 0.5% budget."""
    stages = [KeywordStage("heuristic", "rule-word"), KeywordStage("tfidf", "subtle-attack")]
    policy = (StagePolicy("heuristic", may_allow_early=False), StagePolicy("tfidf"))
    cascade = CascadeDetector(stages, policy=policy, block_at=0.995, allow_at=0.90)
    noisy = [f"benign {i}" for i in range(99)] + ["rule-word here"]
    cascade.calibrate(noisy, [0] * 100)
    for stage in stages:
        stage.calls = 0
    cascade.score_one("rule-word appears here")
    assert stages[1].calls == 1, "a 1%-FPR rule must not short-circuit a 0.5% budget"


def test_scoring_before_calibration_is_refused():
    stages = [KeywordStage("heuristic", "x"), KeywordStage("tfidf", "y")]
    cascade = CascadeDetector(stages, policy=(StagePolicy("heuristic"), StagePolicy("tfidf")))
    with pytest.raises(RuntimeError):
        cascade.score(["anything"])


def test_empty_input():
    cascade, _ = _cascade(first_may_allow=False)
    assert len(cascade.score([])) == 0


def test_save_and_load_preserves_the_policy(tmp_path, monkeypatch):
    cascade, stages = _cascade(first_may_allow=False)
    cascade.save(tmp_path / "cascade")

    import garde_fou.evaluation.runner as runner

    by_name = {s.name: s for s in stages}
    monkeypatch.setattr(runner, "load_or_train", lambda name: by_name[name])

    restored = CascadeDetector.load(tmp_path / "cascade")
    assert [p.may_allow_early for p in restored.policy] == [False, True]
    assert restored.score_one("a subtle-attack") > 0.9
