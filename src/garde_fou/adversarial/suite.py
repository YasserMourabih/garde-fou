"""Assemble the two stress-test sets: `test_fr` and `test_adv`."""

from __future__ import annotations

import json
import logging
import random

import pandas as pd

from garde_fou.adversarial.fr_corpus import build_fr_corpus
from garde_fou.adversarial.mutators import MUTATORS
from garde_fou.data.build import load_split
from garde_fou.data.sources import COLUMNS
from garde_fou.paths import PROCESSED, RESOURCES, ensure_dirs

log = logging.getLogger(__name__)

SEED = 11
# Attacks and benign prompts drawn per mutator, so every family gets the same
# sample size and the per-family numbers stay comparable.
N_ATTACKS_PER_MUTATOR = 220
N_BENIGN_PER_MUTATOR = 220
N_FR_BENIGN_TRAFFIC = 2000

OUT_COLS = [*COLUMNS, "family", "eval_set", "mutator", "fidelity", "origin_id"]


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in OUT_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[OUT_COLS]


def build_fr_set(*, seed: int = SEED) -> pd.DataFrame:
    """French attacks + French hard negatives + real French traffic."""
    hand = _frame(build_fr_corpus())
    hand["eval_set"] = "test_fr"
    hand["mutator"] = "none"
    hand["fidelity"] = "high"

    pool = load_split("pool_fr_test")
    traffic = _frame(pool.sample(n=min(N_FR_BENIGN_TRAFFIC, len(pool)), random_state=seed))
    traffic["eval_set"] = "test_fr"
    traffic["family"] = "fr_traffic"
    traffic["mutator"] = "none"
    traffic["fidelity"] = "high"

    return pd.concat([hand, traffic], ignore_index=True).reset_index(drop=True)


def build_adv_set(*, seed: int = SEED) -> pd.DataFrame:
    """Every mutator applied to a fixed sample of held-out attacks *and* benign prompts."""
    rng = random.Random(seed)
    attacks = load_split("pool_attacks").reset_index(drop=True)
    benign = load_split("pool_fr_adv").reset_index(drop=True)

    rows: list[dict[str, object]] = []

    def emit(src: pd.DataFrame, n: int, mutator) -> None:
        idx = rng.sample(range(len(src)), k=min(n, len(src)))
        for i in idx:
            row = src.iloc[i]
            mutated = mutator(str(row["text"]), random.Random(rng.randrange(1 << 30)))
            rows.append(
                {
                    "text": mutated,
                    "label": int(row["label"]),
                    "source": f"{row['source']}+{mutator.name}",
                    "lang": row["lang"],
                    "split": "test",
                    "family": mutator.family,
                    "eval_set": "test_adv",
                    "mutator": mutator.name,
                    "fidelity": mutator.fidelity,
                    "origin_id": f"{row['source']}:{i}",
                }
            )

    for mutator in MUTATORS:
        emit(attacks, N_ATTACKS_PER_MUTATOR, mutator)
        emit(benign, N_BENIGN_PER_MUTATOR, mutator)

    # Unmutated controls from the same pools: the robustness delta is only
    # meaningful against the score on the *very same* prompts untouched.
    for src, n in ((attacks, N_ATTACKS_PER_MUTATOR), (benign, N_BENIGN_PER_MUTATOR)):
        idx = rng.sample(range(len(src)), k=min(n, len(src)))
        for i in idx:
            row = src.iloc[i]
            rows.append(
                {
                    "text": str(row["text"]),
                    "label": int(row["label"]),
                    "source": str(row["source"]),
                    "lang": row["lang"],
                    "split": "test",
                    "family": "control",
                    "eval_set": "test_adv",
                    "mutator": "none",
                    "fidelity": "high",
                    "origin_id": f"{row['source']}:{i}",
                }
            )

    return _frame(rows)


def build_adversarial_sets(*, seed: int = SEED) -> dict[str, pd.DataFrame]:
    ensure_dirs()
    sets = {"test_fr": build_fr_set(seed=seed), "test_adv": build_adv_set(seed=seed)}
    for name, df in sets.items():
        df.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        log.info("wrote %s (%d rows, %d attacks)", name, len(df), int((df.label == 1).sum()))

    # Ship the hand-written French set as a stand-alone, citable artefact.
    RESOURCES.mkdir(parents=True, exist_ok=True)
    hand = sets["test_fr"]
    hand = hand[hand["source"] == "fr_handcrafted"]
    path = RESOURCES / "fr_injection_eval.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in hand[["text", "label", "family"]].to_dict("records"):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("wrote %s (%d rows)", path.name, len(hand))
    return sets
