"""Public Hugging Face sources, mapped onto one schema.

Every corpus is reduced to `text, label, source, lang, split` where
`label == 1` means "prompt injection / jailbreak attempt".

We deliberately read the Hub's *auto-converted parquet* endpoint with the
standard library instead of depending on `datasets`: it keeps the dependency
tree small, the download is a single cached file per split, and the exact
bytes we trained on stay reproducible.
"""

from __future__ import annotations

import io
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from garde_fou import BENIGN, INJECTION
from garde_fou.paths import RAW

log = logging.getLogger(__name__)

PARQUET_API = "https://huggingface.co/api/datasets/{repo}/parquet"
USER_AGENT = "garde-fou/0.1 (+https://github.com/)"
COLUMNS = ["text", "label", "source", "lang", "split"]


@dataclass(frozen=True)
class Source:
    """A public corpus plus the few facts needed to normalise it."""

    name: str
    repo_id: str
    text_col: str
    lang: str
    role: str  # "attack", "benign" or "mixed"
    note: str
    splits: tuple[str, ...] = ("train",)
    label_col: str | None = None
    label_map: dict[str, int] = field(default_factory=dict)
    fixed_label: int | None = None
    max_rows: int | None = None


SOURCES: dict[str, Source] = {
    "safeguard": Source(
        name="safeguard",
        repo_id="xTRam1/safe-guard-prompt-injection",
        splits=("train", "test"),
        text_col="text",
        label_col="label",
        lang="en",
        role="mixed",
        note="Largest public injection/benign mix. Used as the training distribution.",
    ),
    "deepset": Source(
        name="deepset",
        repo_id="deepset/prompt-injections",
        splits=("train", "test"),
        text_col="text",
        label_col="label",
        lang="en+de",
        role="mixed",
        note="Independently collected, partly German. Held out as out-of-distribution.",
    ),
    "jailbreak": Source(
        name="jailbreak",
        repo_id="jackhhao/jailbreak-classification",
        splits=("train", "test"),
        text_col="prompt",
        label_col="type",
        label_map={"benign": BENIGN, "jailbreak": INJECTION},
        lang="en",
        role="mixed",
        note="Long role-play jailbreaks (DAN family). Held out as out-of-distribution.",
    ),
    "gandalf": Source(
        name="gandalf",
        repo_id="Lakera/gandalf_ignore_instructions",
        splits=("train", "validation", "test"),
        text_col="text",
        fixed_label=INJECTION,
        lang="en",
        role="attack",
        note="Real attacks typed by humans against the Gandalf game. Attacks only.",
    ),
    "alpaca_en": Source(
        name="alpaca_en",
        repo_id="tatsu-lab/alpaca",
        text_col="instruction",
        fixed_label=BENIGN,
        lang="en",
        role="benign",
        note="Stand-in for ordinary English user traffic.",
        max_rows=6000,
    ),
    "alpaca_fr": Source(
        name="alpaca_fr",
        repo_id="jpacifico/French-Alpaca-dataset-Instruct-110K",
        text_col="instruction",
        fixed_label=BENIGN,
        lang="fr",
        role="benign",
        note="Stand-in for ordinary French user traffic. Never seen during training.",
        max_rows=6000,
    ),
}


def _fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    log.info("downloading %s", url)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read()
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"could not download {url}: {exc}") from exc
    dest.write_bytes(payload)
    return dest


def _parquet_index(src: Source) -> dict[str, list[str]]:
    idx_path = RAW / src.name / "parquet_index.json"
    _fetch(PARQUET_API.format(repo=src.repo_id), idx_path)
    index = json.loads(idx_path.read_text())
    # {config: {split: [urls]}} -> keep the first (and in practice only) config.
    config = next(iter(index.values()))
    return config


def load_source(src: Source | str, *, splits: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Download (once) and normalise one corpus."""
    if isinstance(src, str):
        src = SOURCES[src]
    index = _parquet_index(src)
    frames: list[pd.DataFrame] = []
    for split in splits or src.splits:
        urls = index.get(split)
        if not urls:
            log.warning("%s has no split %r (available: %s)", src.name, split, list(index))
            continue
        for shard, url in enumerate(urls):
            path = _fetch(url, RAW / src.name / f"{split}-{shard}.parquet")
            raw = pd.read_parquet(io.BytesIO(path.read_bytes()))
            frames.append(_normalise(raw, src, split))
    if not frames:
        raise RuntimeError(f"no data recovered for {src.repo_id}")
    out = pd.concat(frames, ignore_index=True)
    if src.max_rows is not None and len(out) > src.max_rows:
        out = out.sample(n=src.max_rows, random_state=0).reset_index(drop=True)
    return out


def _normalise(raw: pd.DataFrame, src: Source, split: str) -> pd.DataFrame:
    if src.text_col not in raw.columns:
        raise KeyError(f"{src.repo_id}:{split} has no column {src.text_col!r} ({list(raw.columns)})")
    df = pd.DataFrame({"text": raw[src.text_col].astype("string")})
    if src.fixed_label is not None:
        df["label"] = src.fixed_label
    else:
        values = raw[src.label_col]
        if src.label_map:
            df["label"] = values.astype("string").str.strip().str.lower().map(src.label_map)
        else:
            df["label"] = pd.to_numeric(values, errors="coerce")
    df["source"] = src.name
    df["lang"] = src.lang
    df["split"] = split
    return clean(df)


def clean(df: pd.DataFrame, *, min_chars: int = 8, max_chars: int = 4000) -> pd.DataFrame:
    """Drop unusable rows, truncate very long ones, de-duplicate."""
    from garde_fou.textnorm import normalize

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= min_chars]
    df["text"] = df["text"].str.slice(0, max_chars)
    # De-duplicate on the normalised form so that trivial re-spacings of the
    # same prompt cannot appear on both sides of a split.
    key = df["text"].map(lambda t: normalize(t)[:400])
    df = df[~key.duplicated()]
    return df[COLUMNS].reset_index(drop=True)
