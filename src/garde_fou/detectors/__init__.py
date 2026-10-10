"""Detector registry.

Two maps, because construction and loading are different needs: `REGISTRY`
holds factories that build a *fresh* detector, while `CLASSES` holds the types
whose `.load()` reads a saved artefact. Keeping them apart matters for the
cascade, whose factory calibrates itself against the validation split — not
something you want to trigger just to look up a class.
"""

from __future__ import annotations

from collections.abc import Callable

from garde_fou.detectors.base import Detector
from garde_fou.detectors.heuristic import HeuristicDetector

REGISTRY: dict[str, Callable[[], Detector]] = {
    "heuristic": HeuristicDetector,
    "heuristic_raw": lambda: HeuristicDetector(normalize_text=False),
}
CLASSES: dict[str, type] = {
    "heuristic": HeuristicDetector,
    "heuristic_raw": HeuristicDetector,
}


def _register_optional() -> None:
    """Add detectors whose dependencies may not be installed."""
    try:
        from garde_fou.detectors.tfidf import TfidfDetector

        REGISTRY["tfidf"] = TfidfDetector
        CLASSES["tfidf"] = TfidfDetector
    except ImportError:  # pragma: no cover - `pip install .[neural]` not run
        pass
    try:
        from garde_fou.detectors.embedding import EmbeddingDetector
        from garde_fou.detectors.transformer import TransformerDetector

        REGISTRY["embedding"] = EmbeddingDetector
        REGISTRY["transformer"] = TransformerDetector
        CLASSES["embedding"] = EmbeddingDetector
        CLASSES["transformer"] = TransformerDetector
    except ImportError:  # pragma: no cover
        pass
    try:
        from garde_fou.detectors.cascade import CascadeDetector

        REGISTRY["cascade"] = CascadeDetector.assemble
        CLASSES["cascade"] = CascadeDetector
    except ImportError:  # pragma: no cover
        pass


_register_optional()


def build(name: str) -> Detector:
    """Construct a fresh detector (the cascade also calibrates itself here)."""
    if name not in REGISTRY:
        raise KeyError(f"unknown detector {name!r} (available: {sorted(REGISTRY)})")
    return REGISTRY[name]()


def detector_class(name: str) -> type:
    if name not in CLASSES:
        raise KeyError(f"unknown detector {name!r} (available: {sorted(CLASSES)})")
    return CLASSES[name]


def available() -> list[str]:
    return sorted(REGISTRY)


__all__ = ["Detector", "REGISTRY", "CLASSES", "build", "detector_class", "available"]
