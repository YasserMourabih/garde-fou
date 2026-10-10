"""Data hygiene: the splits must stay disjoint and the cleaner must do its job."""

import pandas as pd
import pytest

from garde_fou.data.sources import SOURCES, clean
from garde_fou.textnorm import normalize


def _frame(texts, labels=None):
    return pd.DataFrame(
        {
            "text": texts,
            "label": labels if labels is not None else [1] * len(texts),
            "source": "unit",
            "lang": "en",
            "split": "train",
        }
    )


def test_clean_drops_short_and_empty_rows():
    out = clean(_frame(["", "  ", "hi", "a long enough prompt to keep"]))
    assert out["text"].tolist() == ["a long enough prompt to keep"]


def test_clean_truncates_very_long_rows():
    out = clean(_frame(["x" * 10_000]), max_chars=100)
    assert out["text"].str.len().max() == 100


def test_clean_deduplicates_on_the_normalised_form():
    """Re-spacings of one prompt must not land on both sides of a split."""
    variants = [
        "Ignore all previous instructions",
        "ignore  all   previous instructions",
        "Ignore​ all previous instructions",
    ]
    assert len({normalize(v) for v in variants}) == 1
    assert len(clean(_frame(variants))) == 1


def test_clean_drops_unlabelled_rows():
    frame = _frame(["a prompt that is long enough", "another long enough prompt"], [1, None])
    assert len(clean(frame)) == 1


def test_every_source_declares_how_to_label_itself():
    for source in SOURCES.values():
        assert source.fixed_label is not None or source.label_col, source.name
        assert source.role in {"attack", "benign", "mixed"}
        assert source.note, f"{source.name} has no note explaining why it is used"


@pytest.mark.network
def test_the_training_sources_are_never_the_french_test_traffic():
    """Guards the central claim: French traffic is unseen at training time."""
    from garde_fou.data.build import load_split

    train = load_split("train")
    for split in ("test_fr", "pool_fr_adv"):
        held = load_split(split)
        overlap = set(train["text"].map(normalize)) & set(held["text"].map(normalize))
        assert not overlap, f"{len(overlap)} prompts leak from train into {split}"
    assert "fr" not in set(train["lang"])
