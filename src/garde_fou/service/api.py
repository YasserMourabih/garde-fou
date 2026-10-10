"""The inference service.

One endpoint that matters, `POST /v1/scan`, and a response shaped so that the
caller can act on it without reading the paper: a verdict, a calibrated score,
the rules that fired, and how much legitimate traffic the chosen threshold
sacrifices. `threshold` is expressed in the same calibrated unit everywhere, so
`threshold=0.99` means "accept blocking 1% of legitimate requests" regardless
of which stage produced the score.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from garde_fou import __version__
from garde_fou.paths import ARTIFACTS

log = logging.getLogger(__name__)

MAX_CHARS = 20_000
MAX_BATCH = 64
DEFAULT_THRESHOLD = float(os.environ.get("GARDE_FOU_THRESHOLD", "0.99"))


class ScanRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0)


class BatchScanRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0)


class ScanResponse(BaseModel):
    blocked: bool
    score: float
    threshold: float
    # Null when the detector could not be calibrated (no validation data on
    # disk): the threshold is then a raw score and carries no FPR guarantee.
    calibrated: bool = True
    legit_traffic_blocked_at_threshold: float | None
    stages_used: list[str]
    reasons: list[dict]
    latency_ms: float
    detector: str


@lru_cache(maxsize=1)
def get_detector():
    """Prefer the calibrated cascade, then a calibrated rule engine, then raw rules.

    The API must boot on a fresh clone with no artefacts, but it must never
    claim an FPR guarantee it cannot back — hence the explicit `calibrated`
    flag on the way down.
    """
    from garde_fou.detectors.calibration import CalibratedDetector
    from garde_fou.detectors.cascade import CascadeDetector
    from garde_fou.detectors.heuristic import HeuristicDetector

    path = ARTIFACTS / "cascade"
    if path.exists():
        try:
            log.info("loading cascade from %s", path)
            return CascadeDetector.load(path)
        except Exception as exc:  # pragma: no cover - artefact mismatch
            log.warning("cascade unusable (%s), falling back", exc)
    try:
        return CascadeDetector.assemble()
    except Exception as exc:
        log.warning("cannot assemble cascade (%s)", exc)

    rules = CalibratedDetector(HeuristicDetector())
    try:
        from garde_fou.data.build import load_split

        val = load_split("val")
        rules.calibrate(val["text"].tolist(), val["label"].tolist())
        log.info("serving the rule engine, calibrated on the validation split")
    except Exception as exc:  # pragma: no cover - fresh clone, no data
        log.warning("no validation data (%s): scores are uncalibrated", exc)
    return rules


WARMUP_PROBE = "warmup probe: what is the capital of Peru?"


def warm_up(detector) -> list[str]:
    """Load every stage's model before the service accepts traffic.

    The encoder stages build their model lazily on first use, so without this
    the first request that happens to escalate pays the whole model load:
    measured at 13.4 s against 13 ms once warm. A guardrail sits in front of
    every request, so being the slowest thing in the path exactly once — for
    one arbitrary user — is not an acceptable trade.

    Each stage is warmed directly rather than by sending a probe through the
    cascade, because a probe that exits early would leave the later stages
    exactly as cold as before.
    """
    warmed = []
    for stage in getattr(detector, "stages", None) or [detector]:
        name = getattr(stage, "name", type(stage).__name__)
        try:
            stage.score([WARMUP_PROBE])
            warmed.append(name)
        except Exception as exc:  # pragma: no cover - a stage may be unusable
            log.warning("could not warm %s: %s", name, exc)
    try:
        detector.score_one(WARMUP_PROBE)
    except Exception as exc:  # pragma: no cover
        log.warning("could not warm the full path: %s", exc)
    return warmed


@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.perf_counter()
    warmed = warm_up(get_detector())
    log.info(
        "warmed %s in %.1f s", ", ".join(warmed) or "nothing", time.perf_counter() - start
    )
    yield


def _scan(detector, text: str, threshold: float) -> ScanResponse:
    start = time.perf_counter()
    if hasattr(detector, "decision"):
        verdict = detector.decision(text, threshold=threshold)
    else:
        detail = detector.explain(text)
        verdict = {
            "blocked": detail["score"] >= threshold,
            "score": detail["score"],
            "calibrated": False,
            "legit_traffic_blocked_at_threshold": None,
            "stages_used": [detector.name],
            "reasons": detail.get("reasons", []),
        }
    verdict.setdefault("calibrated", True)
    return ScanResponse(
        **verdict,
        threshold=threshold,
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
        detector=detector.name,
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Garde-Fou",
        version=__version__,
        summary="Multilingual prompt-injection screening for LLM applications.",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/config")
    def config() -> dict:
        detector = get_detector()
        return {
            "detector": detector.name,
            "stages": getattr(detector, "stage_names", [detector.name]),
            "default_threshold": DEFAULT_THRESHOLD,
            "calibrated": bool(getattr(detector, "calibrated", True)),
            "threshold_semantics": (
                "Scores are FPR-calibrated on held-out benign traffic: a threshold of t "
                "blocks about (1 - t) of legitimate requests."
            ),
        }

    @app.post("/v1/scan", response_model=ScanResponse)
    def scan(req: ScanRequest) -> ScanResponse:
        return _scan(get_detector(), req.text, req.threshold)

    @app.post("/v1/scan/batch", response_model=list[ScanResponse])
    def scan_batch(req: BatchScanRequest) -> list[ScanResponse]:
        if any(len(t) > MAX_CHARS for t in req.texts):
            raise HTTPException(status_code=422, detail=f"each text must be <= {MAX_CHARS} chars")
        detector = get_detector()
        return [_scan(detector, t, req.threshold) for t in req.texts]

    return app


app = create_app()
