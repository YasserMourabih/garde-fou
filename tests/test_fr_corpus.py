"""The French set is a deliverable in its own right, so it gets its own checks."""

from garde_fou import BENIGN, INJECTION
from garde_fou.adversarial.fr_corpus import HARD_NEGATIVES, build_fr_corpus


def test_corpus_has_both_classes_in_useful_quantities():
    rows = build_fr_corpus()
    attacks = [r for r in rows if r["label"] == INJECTION]
    benign = [r for r in rows if r["label"] == BENIGN]
    assert len(attacks) >= 150
    assert len(benign) >= 40


def test_no_duplicate_prompts():
    rows = build_fr_corpus()
    keys = [" ".join(str(r["text"]).split()).lower() for r in rows]
    assert len(keys) == len(set(keys))


def test_deterministic():
    assert build_fr_corpus(seed=7) == build_fr_corpus(seed=7)


STOPWORDS = {
    "dans", "pour", "avec", "sans", "cette", "votre", "notre", "plus", "tout",
    "tous", "toute", "toutes", "mais", "comme", "quoi", "quel", "quelle",
    "quelles", "quels", "dont", "donc", "elle", "nous", "vous", "leur", "leurs",
    "peux", "comment", "pourquoi", "cela", "très", "bien", "aussi", "autre",
    "autres", "entre", "chaque", "fait", "faire", "suis", "sont", "était",
    "avant", "après", "même", "alors", "ainsi", "puis", "cique",
}

# Traps that are structural rather than lexical: the giveaway is a *pattern*
# shared with an attack, not a word. Role assignment and self-policy probing
# are exactly the shapes a naive filter over-blocks.
STRUCTURAL_TRAPS = (
    "tu es ",          # role assignment — identical in shape to a persona jailbreak
    "le droit",        # probing the assistant's own policy
    "tes limites",
)
STEM = 5


def _stems(text: str) -> set[str]:
    import re

    words = re.split(r"[^a-zà-ÿ]+", text.lower())
    return {w[:STEM] for w in words if len(w) >= 4 and w not in STOPWORDS}


def _attack_stems() -> set[str]:
    stems: set[str] = set()
    for row in build_fr_corpus():
        if row["label"] == INJECTION:
            stems |= _stems(str(row["text"]))
    return stems


def test_hard_negatives_look_like_attacks():
    """Every hard negative must overlap an attack lexically or structurally.

    Deriving the trap set from the corpus instead of hand-listing it means the
    test keeps its meaning as the corpus grows: a hard negative that resembles
    no attack at all is not hard, it is just another benign prompt, and it would
    quietly stop testing anything. Stems rather than whole words, because
    `désactiver` and `désactive` are the same trap.
    """
    vocabulary = _attack_stems()
    toothless = [
        text
        for text in HARD_NEGATIVES
        if not (_stems(text) & vocabulary)
        and not any(trap in text.lower() for trap in STRUCTURAL_TRAPS)
    ]
    assert not toothless, f"these hard negatives resemble no attack: {toothless}"


def test_hard_negatives_are_not_accidentally_attacks():
    """Sanity check on the labels: no hard negative may pair an override opener
    with an extraction payload, which is what makes a prompt an attack here."""
    from garde_fou.adversarial.fr_corpus import OPENERS, PAYLOADS

    for text in HARD_NEGATIVES:
        assert text not in OPENERS
        assert text not in PAYLOADS


def test_every_row_has_the_expected_schema():
    for row in build_fr_corpus():
        assert set(row) == {"text", "label", "source", "lang", "split", "family"}
        assert row["lang"] == "fr"
        assert row["label"] in {BENIGN, INJECTION}
        assert str(row["text"]).strip()
