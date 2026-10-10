"""Build the evaluation matrix.

The whole point of this project is that *where* you test decides what you
conclude. So the splits are explicit and deliberately uneven:

| set        | what it answers                                        |
|------------|--------------------------------------------------------|
| `train`    | fitting distribution: one public corpus + English traffic |
| `val`      | threshold calibration only — never reported            |
| `test_in`  | in-distribution score (the number papers usually quote) |
| `test_ood` | does it transfer to corpora collected by other people?  |
| `test_fr`  | does it survive a language absent from training?        |
| `test_adv` | does it survive cheap obfuscation and re-framing?       |

`train` is intentionally restricted to `safeguard` + English `alpaca` so that
every other corpus stays genuinely unseen.
"""

from __future__ import annotations

import json
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from garde_fou.data.sources import COLUMNS, load_source
from garde_fou.paths import PROCESSED, ensure_dirs

log = logging.getLogger(__name__)

SEED = 42
VAL_FRACTION = 0.15

# `alpaca_fr` is the only French traffic we have, and three different eval sets
# need some. We slice it once, disjointly, so no French prompt is ever scored
# twice under two different headings.
N_FR_OOD = 2500   # -> test_ood
N_FR_ADV = 1500   # -> mutated benign in test_adv
N_FR_TEST = 2000  # -> untouched French traffic in test_fr


def _tag(df: pd.DataFrame, eval_set: str) -> pd.DataFrame:
    out = df.copy()
    out["eval_set"] = eval_set
    out["family"] = out.get("family", "natural")
    return out[[*COLUMNS, "family", "eval_set"]]


def build_splits(*, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Materialise every split under `data/processed/`."""
    ensure_dirs()

    safeguard = load_source("safeguard")
    deepset = load_source("deepset")
    jailbreak = load_source("jailbreak")
    gandalf = load_source("gandalf")
    alpaca_en = load_source("alpaca_en")
    alpaca_fr = load_source("alpaca_fr")

    # ---- training distribution -------------------------------------------
    fit_pool = pd.concat(
        [safeguard[safeguard.split == "train"], alpaca_en],
        ignore_index=True,
    )
    train, val = train_test_split(
        fit_pool,
        test_size=VAL_FRACTION,
        random_state=seed,
        stratify=fit_pool["label"],
    )

    # ---- in-distribution test --------------------------------------------
    test_in = safeguard[safeguard.split == "test"]

    # ---- held-out French traffic, split between two consumers ------------
    fr = alpaca_fr.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    a, b = N_FR_OOD, N_FR_OOD + N_FR_ADV
    fr_ood, fr_adv, fr_test = fr.iloc[:a], fr.iloc[a:b], fr.iloc[b : b + N_FR_TEST]

    # ---- cross-dataset test ----------------------------------------------
    test_ood = pd.concat(
        [deepset, jailbreak[jailbreak.split == "test"], gandalf, fr_ood],
        ignore_index=True,
    )

    splits = {
        "train": _tag(train, "train"),
        "val": _tag(val, "val"),
        "test_in": _tag(test_in, "test_in"),
        "test_ood": _tag(test_ood, "test_ood"),
    }

    # ---- pools consumed by the adversarial suite -------------------------
    pools = {
        "pool_attacks": _tag(test_ood[test_ood.label == 1], "pool_attacks"),
        "pool_fr_adv": _tag(fr_adv, "pool_fr_adv"),
        "pool_fr_test": _tag(fr_test, "pool_fr_test"),
    }

    for name, df in {**splits, **pools}.items():
        path = PROCESSED / f"{name}.parquet"
        df.to_parquet(path, index=False)
        log.info("wrote %s (%d rows)", path.name, len(df))

    _write_manifest({**splits, **pools})
    return splits


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `garde-fou data build` first")
    return pd.read_parquet(path)


def _write_manifest(splits: dict[str, pd.DataFrame]) -> None:
    manifest = {
        name: {
            "rows": int(len(df)),
            "injection": int((df.label == 1).sum()),
            "benign": int((df.label == 0).sum()),
            "sources": sorted(df["source"].unique().tolist()),
            "langs": sorted(df["lang"].unique().tolist()),
        }
        for name, df in splits.items()
    }
    (PROCESSED / "manifest.json").write_text(json.dumps(manifest, indent=2))


def summary_table(splits: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    names = ["train", "val", "test_in", "test_ood", "test_fr", "test_adv"]
    rows = []
    for name in names:
        try:
            df = splits[name] if splits and name in splits else load_split(name)
        except (FileNotFoundError, KeyError):
            continue
        rows.append(
            {
                "set": name,
                "rows": len(df),
                "injection": int((df.label == 1).sum()),
                "benign": int((df.label == 0).sum()),
                "sources": ", ".join(sorted(df["source"].unique())),
            }
        )
    return pd.DataFrame(rows)
