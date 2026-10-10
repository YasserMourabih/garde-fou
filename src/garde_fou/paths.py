"""Single source of truth for where things live on disk."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RESOURCES = Path(__file__).resolve().parent / "resources"


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, ARTIFACTS, REPORTS, FIGURES, RESOURCES):
        p.mkdir(parents=True, exist_ok=True)
