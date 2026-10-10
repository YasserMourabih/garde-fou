"""Latency, measured the way a guardrail is actually called.

Batch size 1, one request at a time, on CPU. A guardrail runs *before* the
model the user is waiting for, so its cost is pure added latency on every
request — including the 99% that are benign. Throughput numbers from batched
GPU inference describe an offline scan, not an inline filter.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from garde_fou.data.build import load_split
from garde_fou.detectors.base import Detector
from garde_fou.paths import REPORTS

log = logging.getLogger(__name__)


def measure(detector: Detector, texts: list[str], *, warmup: int = 5) -> dict:
    for t in texts[:warmup]:
        detector.score_one(t)
    timings = []
    for t in texts:
        start = time.perf_counter()
        detector.score_one(t)
        timings.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(timings)
    return {
        "n": len(arr),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
    }


def benchmark(detector_names: tuple[str, ...], *, n: int = 150, seed: int = 0) -> pd.DataFrame:
    from garde_fou.evaluation.runner import load_or_train

    pool = load_split("test_ood").sample(n=n, random_state=seed)
    texts = pool["text"].astype(str).tolist()

    rows = []
    for name in detector_names:
        detector = load_or_train(name)
        # Force CPU for the neural detectors: that is the deployment we are costing.
        if hasattr(detector, "device"):
            detector.device = "cpu"
            if getattr(detector, "_model", None) is not None:
                detector._model = detector._model.to("cpu")
        log.info("timing %s over %d single requests", name, len(texts))
        rows.append({"detector": name, "kind": detector.kind, **measure(detector, texts)})

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "latency.csv", index=False)
    log.info("wrote latency.csv")
    return table
